"""
.gauss 파일 포맷 - GMM+ANS 압축 가중치 저장/로드

구조:
  magic[8]       b"GAUSS001"
  n_layers[4]    uint32 LE
  --- 인덱스 (레이어별 메타데이터) ---
  name_len[2], name[n], ndim[1], shape[ndim*4],
  K[1], pi[K*8], mu[K*8], sigma[K*8],
  idx_words[4], resid_words[4]
  --- 데이터 (레이어 순서대로) ---
  idx_data[idx_words*4], resid_data[resid_words*4]
"""

import numpy as np
import struct
from pathlib import Path

MAGIC   = b"GAUSS001"
VERSION = 1


# ── Writer ────────────────────────────────────────────────

class GaussWriter:
    """
    압축 결과를 .gauss 파일로 저장.

    사용법:
        w = GaussWriter("model.gauss")
        w.add(key, shape, K, pi, mu, sigma, idx_compressed, resid_compressed)
        ...
        w.save()
        print(w.summary())
    """

    def __init__(self, path: str):
        self.path   = Path(path)
        self._layers = []  # (key, shape, K, pi, mu, sigma, ic, rc)

    def add(self, key: str, shape: tuple, K: int,
            pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
            ic: np.ndarray, rc: np.ndarray):
        """레이어 하나 추가. ic/rc 는 constriction.get_compressed() 의 uint32 배열."""
        self._layers.append((key, shape, K,
                              np.asarray(pi, np.float64),
                              np.asarray(mu, np.float64),
                              np.asarray(sigma, np.float64),
                              np.asarray(ic, np.uint32),
                              np.asarray(rc, np.uint32)))

    def save(self) -> int:
        """파일 저장. 바이트 크기 반환."""
        with open(self.path, 'wb') as f:
            # 헤더
            f.write(MAGIC)
            f.write(struct.pack('<I', len(self._layers)))

            # 인덱스 섹션
            for key, shape, K, pi, mu, sigma, ic, rc in self._layers:
                nb = key.encode('utf-8')
                f.write(struct.pack('<H', len(nb))); f.write(nb)
                f.write(struct.pack('<B', len(shape)))
                for s in shape: f.write(struct.pack('<I', s))
                f.write(struct.pack('<B', K))
                pi.tofile(f); mu.tofile(f); sigma.tofile(f)
                f.write(struct.pack('<II', len(ic), len(rc)))

            # 데이터 섹션
            for _, _, _, _, _, _, ic, rc in self._layers:
                ic.tofile(f); rc.tofile(f)

        return self.path.stat().st_size

    def summary(self) -> str:
        total_orig = sum(int(np.prod(s)) * 4 for _, s, *_ in self._layers)
        total_comp = self.path.stat().st_size
        return (f"{len(self._layers)} 레이어 | "
                f"원본 {total_orig/1024/1024:.1f} MB → "
                f"압축 {total_comp/1024/1024:.1f} MB | "
                f"비율 {total_orig/total_comp:.3f}x")


# ── Reader ────────────────────────────────────────────────

class GaussReader:
    """
    .gauss 파일에서 레이어 복원.

    사용법:
        r = GaussReader("model.gauss")
        print(r.keys())
        weights = r.decompress("layers.0.ffn.w1.weight")  # np.ndarray f32
        state_dict = r.decompress_all()                    # {key: np.ndarray}
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._index: list = []
        self._data_offset: int = 0
        self._parse_index()

    def _parse_index(self):
        with open(self.path, 'rb') as f:
            magic = f.read(8)
            if magic != MAGIC:
                raise ValueError(f"잘못된 파일: magic={magic!r}")
            n = struct.unpack('<I', f.read(4))[0]

            for _ in range(n):
                name_len = struct.unpack('<H', f.read(2))[0]
                key      = f.read(name_len).decode('utf-8')
                ndim     = struct.unpack('<B', f.read(1))[0]
                shape    = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
                K        = struct.unpack('<B', f.read(1))[0]
                pi       = np.frombuffer(f.read(K * 8), np.float64).copy()
                mu       = np.frombuffer(f.read(K * 8), np.float64).copy()
                sigma    = np.frombuffer(f.read(K * 8), np.float64).copy()
                ic_words, rc_words = struct.unpack('<II', f.read(8))
                self._index.append(dict(
                    key=key, shape=shape, K=K,
                    pi=pi, mu=mu, sigma=sigma,
                    ic_words=ic_words, rc_words=rc_words,
                ))

            self._data_offset = f.tell()

    def keys(self) -> list:
        return [e['key'] for e in self._index]

    def __len__(self) -> int:
        return len(self._index)

    def decompress(self, key: str) -> np.ndarray:
        """단일 레이어 복원 → float32 numpy 배열 (원본 shape)."""
        import constriction
        from gmm_compress import RESID_SCALE, RESID_BOUND

        e = next((x for x in self._index if x['key'] == key), None)
        if e is None:
            raise KeyError(key)

        # 데이터 섹션에서 offset 계산
        offset = self._data_offset
        for x in self._index:
            if x['key'] == key:
                break
            offset += (x['ic_words'] + x['rc_words']) * 4

        with open(self.path, 'rb') as f:
            f.seek(offset)
            ic = np.frombuffer(f.read(e['ic_words'] * 4), np.uint32).copy()
            rc = np.frombuffer(f.read(e['rc_words'] * 4), np.uint32).copy()

        N = int(np.prod(e['shape']))
        pi, mu, sigma = e['pi'], e['mu'], e['sigma']

        # 인덱스 복원
        probs = (pi / pi.sum()).astype(np.float32)
        model = constriction.stream.model.Categorical(probs, perfect=False)
        ans   = constriction.stream.stack.AnsCoder(ic)
        asgn  = ans.decode(model, N)

        # 잔차 복원
        sigma_a = sigma[asgn]
        means   = np.zeros(N, np.float64)
        stds    = np.maximum(sigma_a * RESID_SCALE, 0.5)
        model_g = constriction.stream.model.QuantizedGaussian(-RESID_BOUND, RESID_BOUND)
        ans2    = constriction.stream.stack.AnsCoder(rc)
        resid_q = ans2.decode(model_g, means, stds)

        weights = (mu[asgn] + resid_q / RESID_SCALE).astype(np.float32)
        return weights.reshape(e['shape'])

    def decompress_all(self) -> dict:
        """모든 레이어 복원 → {key: np.ndarray}."""
        return {e['key']: self.decompress(e['key']) for e in self._index}


# ── CLI: 압축 / 복원 ─────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys, time
    from safetensors.torch import load_file, save_file
    import torch

    parser = argparse.ArgumentParser()
    sub    = parser.add_subparsers(dest='cmd')

    # compress 서브커맨드
    p_c = sub.add_parser('compress', help='.safetensors → .gauss')
    p_c.add_argument('input',  help='입력 .safetensors')
    p_c.add_argument('output', help='출력 .gauss')
    p_c.add_argument('--K',        type=int, default=16)
    p_c.add_argument('--max-iter', type=int, default=100)
    p_c.add_argument('--workers',  type=int, default=None)

    # decompress 서브커맨드
    p_d = sub.add_parser('decompress', help='.gauss → .safetensors')
    p_d.add_argument('input',  help='입력 .gauss')
    p_d.add_argument('output', help='출력 .safetensors')

    # info 서브커맨드
    p_i = sub.add_parser('info', help='.gauss 파일 정보')
    p_i.add_argument('file')

    args = parser.parse_args()

    if args.cmd == 'info':
        r = GaussReader(args.file)
        size = Path(args.file).stat().st_size
        print(f"파일:    {args.file}")
        print(f"크기:    {size/1024/1024:.2f} MB")
        print(f"레이어:  {len(r)} 개")
        for e in r._index:
            n = int(np.prod(e['shape']))
            comp = (e['ic_words'] + e['rc_words']) * 4
            print(f"  {e['key']:<50}  {n*4//1024:>6}KB → {comp//1024:>5}KB  "
                  f"{n*4/comp:.3f}x  K={e['K']}")

    elif args.cmd == 'compress':
        from multiprocessing import cpu_count
        from gmm_compress import _worker

        print(f"Loading {args.input} ...")
        sd = load_file(args.input)
        keys = sorted(
            [k for k, v in sd.items() if v.numel() >= 1000],
            key=lambda k: sd[k].numel(), reverse=True
        )
        n_workers = args.workers or cpu_count()
        tasks = [
            (k, sd[k].float().numpy().flatten(),
             args.K, args.max_iter, 200_000, False)
            for k in keys
        ]
        print(f"레이어 {len(keys)}개, K={args.K}, workers={n_workers}")

        from multiprocessing import Pool
        writer = GaussWriter(args.output)
        t0     = time.perf_counter()

        with Pool(n_workers) as pool:
            # _worker가 ic/rc를 반환하도록 수정 필요 → 아래 _worker_with_data 사용
            pass

        # _worker_with_data: ic, rc 원본 반환 버전
        def _worker_save(args_):
            key, data_np, K, max_iter, max_fit, _ = args_
            from gmm_compress import (fit_gmm_fast, assign_all,
                                          encode_indices, encode_residuals,
                                          RESID_SCALE, RESID_BOUND)
            import traceback, numpy as np, time
            try:
                data = data_np.astype(np.float64)
                N    = len(data)
                t0   = time.perf_counter()
                fit_data = data if N <= max_fit else data[
                    np.random.choice(N, max_fit, replace=False)]
                pi, mu, sigma, n_iter = fit_gmm_fast(fit_data, K, max_iter)
                asgn    = assign_all(data, pi, mu, sigma)
                mu_a    = mu[asgn]; sigma_a = sigma[asgn]
                resid_q = np.clip(
                    np.round((data - mu_a) * RESID_SCALE).astype(np.int32),
                    -RESID_BOUND, RESID_BOUND)
                ic = encode_indices(asgn, pi)
                rc = encode_residuals(resid_q, sigma_a)
                orig = N * 4
                comp = (len(ic) + len(rc)) * 4 + K * 3 * 8
                return {'key': key, 'pi': pi, 'mu': mu, 'sigma': sigma,
                        'ic': ic, 'rc': rc, 'K': K,
                        'orig': orig, 'comp': comp,
                        'ratio': orig/comp, 'n_iter': n_iter,
                        't': time.perf_counter()-t0, 'error': None}
            except Exception:
                return {'key': key, 'error': traceback.format_exc()}

        shapes = {k: tuple(sd[k].shape) for k in keys}
        total_orig = total_comp = 0

        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker_save, tasks), 1):
                if res['error']:
                    print(f"  ERROR [{res['key']}]"); continue
                writer.add(res['key'], shapes[res['key']], res['K'],
                           res['pi'], res['mu'], res['sigma'],
                           res['ic'], res['rc'])
                total_orig += res['orig']; total_comp += res['comp']
                print(f"  [{i:4d}/{len(tasks)}] {res['key']:<48}"
                      f"  {res['ratio']:.3f}x  iter={res['n_iter']:3d}"
                      f"  {res['t']:.1f}s")

        saved = writer.save()
        elapsed = time.perf_counter() - t0
        print(f"\n저장: {args.output}  ({saved/1024/1024:.1f} MB)")
        print(f"전체 압축률: {total_orig/total_comp:.3f}x  "
              f"({total_orig//1024//1024}MB → {saved//1024//1024}MB)  "
              f"총 {elapsed:.0f}초")

    elif args.cmd == 'decompress':
        print(f"Loading {args.input} ...")
        r  = GaussReader(args.input)
        t0 = time.perf_counter()
        sd = {}
        for i, key in enumerate(r.keys(), 1):
            arr = r.decompress(key)
            sd[key] = torch.from_numpy(arr)
            print(f"  [{i:4d}/{len(r)}] {key}")
        save_file(sd, args.output)
        print(f"\n저장: {args.output}  ({Path(args.output).stat().st_size/1024/1024:.1f} MB)")
        print(f"복원 완료  ({time.perf_counter()-t0:.1f}초)")

    else:
        parser.print_help()
