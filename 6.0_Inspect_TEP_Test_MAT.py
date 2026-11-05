"""Inspect MATLAB TEP testing file layout before converting to parquet.

Read-only: prints MAT variable names, shapes, dtypes, and a small metadata preview.
Supports MATLAB <=v7.2 (scipy) and HDF5 v7.3 (h5py).
Does not load full arrays or modify source data.
"""
from pathlib import Path
import argparse

def _describe_scipy(path):
    import scipy.io as sio
    print("MAT reader: scipy.io.whosmat (MAT <= v7.2)",flush=True)
    for name,shape,kind in sio.whosmat(str(path)):
        print(f"  variable={name!r} shape={shape} class={kind}",flush=True)

def _describe_h5(path):
    import h5py
    print("MAT reader: h5py (HDF5 / MATLAB v7.3)",flush=True)
    with h5py.File(path,"r") as f:
        def visit(name,obj):
            if name.startswith("#refs#"):
                return
            if isinstance(obj,h5py.Dataset):
                cls=obj.attrs.get("MATLAB_class",b"")
                if isinstance(cls,bytes):
                    cls=cls.decode(errors="replace")
                print(f"  dataset={name!r} shape={obj.shape} dtype={obj.dtype} MATLAB_class={cls}",flush=True)
            else:
                print(f"  group={name!r}",flush=True)
        f.visititems(visit)

def main():
    p=argparse.ArgumentParser(description="Inspect MATLAB testing data variables without conversion")
    p.add_argument("--dir",default="/content/drive/MyDrive/Paper 5",help="Drive folder containing testing .mat file")
    p.add_argument("--filename",default=None,help="Exact .mat filename if multiple candidates exist")
    p.add_argument("--path",default=None,help="Exact full path to one MAT file (avoids same-named files)")
    args=p.parse_args()
    folder=Path(args.dir)
    if args.path:
        mat=Path(args.path)
        if not mat.is_file() or mat.suffix.lower() != ".mat":
            raise FileNotFoundError(f"Not a readable .mat file: {mat}")
        print(f"INSPECTING EXACT FILE: {mat}",flush=True)
        try:
            _describe_scipy(mat)
        except (NotImplementedError,ValueError,TypeError) as err:
            print(f"scipy cannot read layout: {err}; trying h5py",flush=True)
            _describe_h5(mat)
        print("\\nNo parquet files created.",flush=True)
        return
    if not folder.is_dir():
        raise FileNotFoundError(f"Drive folder unavailable: {folder} (check mount or --dir)")
    files=sorted(x for x in folder.rglob("*") if x.is_file() and x.suffix.lower()==".mat")
    print("MAT FILES FOUND",flush=True)
    for x in files:
        print(f"  {x.relative_to(folder)} ({x.stat().st_size/1048576:.1f} MiB)",flush=True)
    if args.filename:
        matches=[x for x in files if x.name.lower()==args.filename.lower()]
    else:
        matches=[x for x in files if "test" in x.stem.lower()]
    if len(matches)!=1:
        print(f"Matching test MAT candidates: {len(matches)}",flush=True)
        print("Specify --filename <exact-name.mat> if needed; no conversion performed.",flush=True)
        return
    mat=matches[0]
    print(f"\nINSPECTING: {mat}",flush=True)
    with mat.open("rb") as f:
        header=f.read(128)
    if header.startswith(b"MATLAB") or header[:4] != b"\x89HDF":
        try:
            _describe_scipy(mat)
            return
        except (NotImplementedError,ValueError,TypeError) as err:
            print(f"scipy could not read layout: {err}; trying h5py",flush=True)
    _describe_h5(mat)
    print("\nNo parquet files created. Review variable shapes before conversion.",flush=True)

if __name__=="__main__":
    main()
