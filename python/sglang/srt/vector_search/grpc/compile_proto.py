#!/usr/bin/env python3
"""
Compile protobuf files for the vector search gRPC service.

This mirrors `python/sglang/srt/grpc/compile_proto.py` but targets
`vector_search.proto` within this directory.
"""

import argparse
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

GRPC_VERSION = "1.75.1"


def get_file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def check_regeneration_needed(proto_file: Path, output_dir: Path) -> bool:
    proto_mtime = get_file_mtime(proto_file)
    generated_files = [
        output_dir / f"{proto_file.stem}_pb2.py",
        output_dir / f"{proto_file.stem}_pb2_grpc.py",
        output_dir / f"{proto_file.stem}_pb2.pyi",
    ]
    return any(get_file_mtime(p) < proto_mtime for p in generated_files)


def fix_imports(output_dir: Path, proto_stem: str) -> None:
    grpc_file = output_dir / f"{proto_stem}_pb2_grpc.py"
    if not grpc_file.exists():
        return
    content = grpc_file.read_text()
    old_import = f"import {proto_stem}_pb2"
    new_import = f"from . import {proto_stem}_pb2"
    if old_import in content:
        grpc_file.write_text(content.replace(old_import, new_import))


def compile_proto(proto_file: Path, output_dir: Path, verbose: bool = True) -> bool:
    if not proto_file.exists():
        print(f"Error: Proto file not found: {proto_file}")
        return False

    try:
        import grpc_tools.protoc  # noqa: F401
    except ImportError:
        print("Error: grpcio-tools not installed")
        print(
            f'Install with: pip install "grpcio-tools=={GRPC_VERSION}" "grpcio=={GRPC_VERSION}"'
        )
        return False

    grpc_tools_version = version("grpcio-tools")
    grpc_version = version("grpcio")
    if grpc_tools_version != GRPC_VERSION or grpc_version != GRPC_VERSION:
        raise RuntimeError(
            f"grpcio-tools {grpc_tools_version} and grpcio {grpc_version} detected; "
            f"{GRPC_VERSION} is required."
        )

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_file.parent}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",
        str(proto_file.name),
    ]
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=proto_file.parent)
    if result.returncode != 0:
        print("Error compiling proto:")
        print(result.stderr)
        if result.stdout:
            print(result.stdout)
        return False

    fix_imports(output_dir, proto_file.stem)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile vector_search.proto")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--proto-file", type=str, default="vector_search.proto")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    proto_file = script_dir / args.proto_file
    output_dir = script_dir
    verbose = not args.quiet

    if args.check:
        if check_regeneration_needed(proto_file, output_dir):
            print("Regeneration needed")
            raise SystemExit(1)
        print("Generated files are up to date")
        return

    ok = compile_proto(proto_file, output_dir, verbose=verbose)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

