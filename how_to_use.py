import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch

from core.opt_rm.model import OptRemoveBertModel
from third.jTrans.data_func import gen_funcstr
import time
import argparse
import contextlib


def cos_similarity_(v1, v2):
	return (v1 @ v2.T / (v1.norm(dim=-1)[:, None] * v2.norm(dim=-1)[None, :]))


def cosine_similarity(v1, v2, device=0):
	# with torch.no_grad():
	#     step1, step2 = 80000, 10000
	#     result = []
	#     for s1 in range(0, len(v1), step1):
	#         res = []
	#         for s2 in range(0, len(v2), step2):
	#             s = cos_similarity_(v1[s1:s1 + step1].clone().to(device), v2[s2:s2 + step2].clone().to(device))
	#             res.append(s.detach().cpu())
	#         result.append(torch.cat(res, dim=1).detach().cpu())
	#     return torch.cat(result, dim=0)
	return cos_similarity_(v1, v2)


def remove_parallel_prefix(checkpoint):
	if 'model' in checkpoint:
		ws = {}
		for k, w in checkpoint['model'].items():
			if k.startswith("module."):
				k = k[len("module."):]
				ws[k] = w
		checkpoint['model'] = ws
	else:
		for mk, mv in checkpoint.items():
			ws = {}
			for k, w in mv.items():
				if k.startswith("module."):
					k = k[len("module."):]
					ws[k] = w
			checkpoint[mk] = ws
	return checkpoint


class FunctionEmbeddingSet(object):
	def __init__(self, opt_to_bin_pkl_paths, limit=None):
		self.func_embeddings, self.func_names = {}, {}
		for opt, bin_pkl_paths in opt_to_bin_pkl_paths.items():
			self.func_embeddings[opt] = []
			self.func_names[opt] = []
			for pkl_path in bin_pkl_paths:
				pkl = pickle.load(open(pkl_path, 'rb'))
				if 'embedding' in pkl:
					self.func_embeddings[opt].append(pkl['embedding'])
					[self.func_names[opt].append((pkl_path, fun_name)) for fun_name in pkl['name']]
				else:
					feats = []
					for offset, (offset, feat, hash, fun_name) in pkl.items():
						feats.append(feat)
						self.func_names[opt].append((pkl_path, fun_name))
					self.func_embeddings[opt].append(torch.stack(feats))
			self.func_embeddings[opt] = torch.cat(self.func_embeddings[opt])

		self.all_func_names = []
		for opt in self.func_embeddings:
			self.all_func_names.extend(self.func_names[opt])

	def query(self, feat, opt=None):
		if opt is None:
			scores = []
			for opt in self.func_embeddings:
				score = cosine_similarity(feat.unsqueeze(dim=0), self.func_embeddings[opt])[0]
				scores.append(score)
			scores = torch.cat(scores)
			max_id = scores.argmax()
			return self.all_func_names[max_id]
		else:
			scores = cosine_similarity(feat.unsqueeze(dim=0), self.func_embeddings[opt])[0]
			max_id = scores.argmax()
			return self.func_names[opt][max_id]

	def __len__(self):
		return len(self.all_func_names)


def _fail(msg: str) -> None:
	"""Fail fast for artifact reproducibility."""
	raise RuntimeError(msg)


def check_state_dict_compat(
	model: torch.nn.Module,
	state: dict,
	*,
	require_no_missing: bool = True,
	suppress_expected_unexpected: bool = True,
) -> None:
	"""
	Validate checkpoint compatibility.

	Policy:
	- Any `missing` keys => fail fast (random/default init would break reproducibility).
	- `unexpected` keys are usually safe, but we optionally allow only a known-safe set.
	"""
	missing, unexpected = model.load_state_dict(state, strict=False)

	allowed_unexpected = {'bert.embeddings.position_ids'} if suppress_expected_unexpected else set()
	unexpected_set = set(unexpected)
	unexpected_bad = sorted(unexpected_set - allowed_unexpected)

	if require_no_missing and len(missing) > 0:
		_fail(
			"Checkpoint/model mismatch: missing keys while loading state_dict.\n"
			f"Missing keys ({len(missing)}): {missing}\n"
			"Missing keys mean some expected parameters/buffers were not loaded and may remain randomly initialized, "
			"which can change inference and break artifact reproducibility."
		)

	if len(unexpected_bad) > 0:
		_fail(
			"Checkpoint/model mismatch: unexpected keys not in allowlist.\n"
			f"Unexpected keys ({len(unexpected_bad)}): {unexpected_bad}\n"
			"If these are not known-safe non-trainable buffers, they may indicate a model-definition mismatch."
		)


def check_position_ids_policy(state: dict) -> None:
	"""
	Position-id buffer policy for reproducibility.
	- Presence in checkpoint is OK; some HF/BERT variants serialize it, some don't.
	- We do NOT fail on its presence. We only use this for cleanliness and to avoid noisy logs.
	"""
	# Known-safe helper buffer; no action needed.
	_ = ('bert.embeddings.position_ids' in state)


def check_pooler_final_matches_checkpoint(
	model_bert: torch.nn.Module,
	state: dict,
) -> None:
	"""
	Ensure the final in-memory pooler weights used for inference match the release checkpoint.
	This guarantees the earlier `from_pretrained(...)` warning does not affect final inference.
	"""
	sd = model_bert.state_dict()

	required = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
	for k in required:
		if k not in sd:
			_fail(f"Pooler sanity check failed: '{k}' missing from final model state_dict.")
		if k not in state:
			_fail(f"Pooler sanity check failed: '{k}' missing from release checkpoint state_dict.")

	# Compare exact tensors (CPU to avoid device mismatch).
	w_ok = torch.allclose(sd['bert.pooler.dense.weight'].cpu(), state['bert.pooler.dense.weight'].cpu())
	b_ok = torch.allclose(sd['bert.pooler.dense.bias'].cpu(), state['bert.pooler.dense.bias'].cpu())

	if not (w_ok and b_ok):
		_fail(
			"Pooler sanity check failed: final pooler params do not match the release checkpoint.\n"
			f"weight_match={w_ok}, bias_match={b_ok}\n"
			"This can affect embeddings because OPTango uses output.pooler_output as fn_embeddings."
		)


@contextlib.contextmanager
def suppress_transformers_warnings(level: str = 'error'):
	"""
	Suppress HuggingFace Transformers log messages (e.g., 'Some weights were not initialized...').
	This does NOT change model parameters; it only reduces noisy logs for artifact reproduction runs.
	"""
	try:
		from transformers.utils import logging as hf_logging
	except Exception:
		# Transformers not available or different layout; do nothing.
		yield
		return

	prev_verbosity = hf_logging.get_verbosity()
	prev_default_handler = hf_logging._default_handler

	# Reduce verbosity and also disable the default handler to avoid stdout spam.
	hf_logging.set_verbosity_error() if level == 'error' else hf_logging.set_verbosity_warning()
	hf_logging.disable_default_handler()

	try:
		yield
	finally:
		# Restore previous logging state.
		hf_logging.set_verbosity(prev_verbosity)
		hf_logging.enable_default_handler()
		# Some versions track handler differently; keep safe behavior.
		hf_logging._default_handler = prev_default_handler


class FullModel(object):
	#! TODO: Fix path
	def __init__(self, device="cuda:0", with_gp=True, checkpoint_dir="model_weight/"):
	#def __init__(self, device="cuda:0", with_gp=True, checkpoint_dir="env/preprocessed/model_weight"):
		checkpoint = torch.load(f"{checkpoint_dir}/model_release.pt", map_location=device)
		checkpoint = remove_parallel_prefix(checkpoint)

		state = checkpoint["bert"]

		with suppress_transformers_warnings(level='error'):
			model = OptRemoveBertModel(feat_source='opt_rm').to(device)

		check_position_ids_policy(state)
		check_state_dict_compat(model, state, require_no_missing=True, suppress_expected_unexpected=True)

		model.eval()
		self.model_bert = model
		check_pooler_final_matches_checkpoint(self.model_bert, state)

		with suppress_transformers_warnings(level='error'):
			model = OptRemoveBertModel(
				feat_source='bsc',
				sub_modules='const_data',
				const_data_kwargs=dict(out_type='const_emb:add'),
			).to(device)
		model.load_state_dict(checkpoint["const"])
		model.eval()
		self.model_const_data = model

		if with_gp:
			with suppress_transformers_warnings(level='error'):
				model = OptRemoveBertModel(feat_source='bsc', sub_modules='group_pred').to(device)
			model.load_state_dict(checkpoint["group_pred"])
			model.eval()
			self.model_group_pred = model
		self.with_gp = with_gp

		self.bsc_feat = torch.zeros(0, ).to(device)  # just for given device, not used
		self.t = {"gp": 0}

	def __call__(self, infos):
		with torch.no_grad():
			feats, other_out = self.model_bert(self.bsc_feat, None, None, infos)
			feats, other_out = self.model_const_data(feats, None, None, infos)
			tic = time.time()
			if self.with_gp:
				feats, other_out = self.model_group_pred(feats, None, None, None)
				other_out["pred_opt"] = [self.model_group_pred.group_predictor.class_names[i] for i in other_out["group_pred_idx"]]
			toc = time.time()
			self.t['gp'] += toc - tic
			return feats, other_out


def load_funcs(ida_path):
	func_names = []
	func_infos = []
	data = pickle.load(open(ida_path, 'rb'))
	for func_name, fun_info in data.items():
		asm_str, asm_info = gen_funcstr(fun_info, convert_jump=True, with_info=True)
		func_names.append(func_name)
		info = {
			"ida_asm_str": asm_str,
			'ida_asm_consts': asm_info['consts'],
		}
		func_infos.append(info)
	return func_names, func_infos


def extract_embeddings(model, func_infos):
	feats, pred_outs = [], []
	for f_info in tqdm(func_infos):
		_feats, other_out = model([f_info])
		pred_out = other_out['pred_opt'][0] if 'pred_opt' in other_out else None
		feats.append(_feats[0].cpu())
		pred_outs.append(pred_out)
	return torch.stack(feats), pred_outs


def matched_gt(func_names1, func_names2):
	match = [-1] * len(func_names1)
	for i1, f1 in enumerate(func_names1):
		for i2, f2 in enumerate(func_names2):
			if f1 == f2:
				match[i1] = i2
				break
	return torch.tensor(match)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--with-gp", default='True')
	args = parser.parse_args()
	print(args)

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	model = FullModel(device, with_gp=eval(args.with_gp))

	# 1. load asm info extracted by ida of two compiler option
	ida_dir = 'data/data-bsca/feat/ida_feat/'
	ida_path = os.path.join(ida_dir, 'openssl', 'gcc-O2', 'openssl_extract.pkl')
	func_names, func_infos = load_funcs(ida_path)
	f1_name, f1_info = func_names[1], func_infos[1]   # get the first function

	ida_path = os.path.join(ida_dir, 'openssl', 'gcc-O3', 'openssl_extract.pkl')
	func_names2, func_infos2 = load_funcs(ida_path)
	f2_names, f2_infos = func_names2[1:10], func_infos2[1:10]   # get first 10 function

	# 2. key step: extract function embedding for given functions
	f1_feat, _ = extract_embeddings(model, [f1_info])
	f2_feats, _ = extract_embeddings(model, f2_infos)

	# 3. calculate similarity score
	similarity = cosine_similarity(f1_feat, f2_feats)
	for i, s in enumerate(similarity[0].cpu().numpy()):
		print(f'cosine similarity({f1_name}, {f2_names[i]}) = {s}')


main()
