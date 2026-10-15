"""Restore large exact-byte result files from the repository's gzip archives."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,tempfile
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    manifest=json.loads((ROOT/'results/stage-b2_3_sync/RAW_ARCHIVES.json').read_text(encoding='utf-8'))
    for item in manifest['archives']:
        archive=(ROOT/item['archive_path']).resolve();target=(ROOT/item['original_path']).resolve()
        if not archive.is_relative_to(ROOT) or not target.is_relative_to(ROOT):raise RuntimeError('Path escaped checkout')
        if sha(archive)!=item['archive_sha256']:raise RuntimeError('Archive checksum mismatch')
        if target.exists():
            if sha(target)!=item['original_sha256']:raise RuntimeError('Existing original differs; refusing overwrite')
            print('EXACT_ORIGINAL_PRESENT '+item['original_path']);continue
        if args.verify_only:
            h=hashlib.sha256();n=0
            with gzip.open(archive,'rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);n+=len(b)
            if h.hexdigest()!=item['original_sha256'] or n!=item['original_bytes']:raise RuntimeError('Decoded checksum mismatch')
            print('ARCHIVE_VERIFIED '+item['original_path']);continue
        descriptor,tmp=tempfile.mkstemp(prefix=target.name+'.restore-',dir=target.parent)
        try:
            with os.fdopen(descriptor,'wb') as out,gzip.open(archive,'rb') as src:
                for b in iter(lambda:src.read(8*1024*1024),b''):out.write(b)
            q=Path(tmp)
            if sha(q)!=item['original_sha256'] or q.stat().st_size!=item['original_bytes']:raise RuntimeError('Decoded checksum mismatch')
            # Creation is atomic, and conflicting pre-existing files are never replaced.
            if os.name=='nt':
                os.rename(q,target)  # Windows rename fails if destination exists.
            else:
                os.link(q,target)    # POSIX atomic create-if-absent.
            print('RESTORED '+item['original_path'])
        finally:Path(tmp).unlink(missing_ok=True)
if __name__=='__main__':main()
