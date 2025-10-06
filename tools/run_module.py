import os
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/run_module.py <module-or-path> -- [args...]")
        sys.exit(2)

    target = sys.argv[1]
    passthrough = []
    if len(sys.argv) > 2 and sys.argv[2] == "--":
        passthrough = sys.argv[3:]
    elif len(sys.argv) > 2:
        passthrough = sys.argv[2:]

    # Project root = parent of this file's directory
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(proj_root, "src")

    # Ensure imports work from both project root (for 'scripts.*') and 'src'
    #  (for 'bank.*')
    for p in (proj_root, src_path):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Run module or path
    if target.endswith(".py") or os.path.sep in target or "/" in target:
        # Treat as a path relative to project root
        script_path = target
        if not os.path.isabs(script_path):
            script_path = os.path.join(proj_root, script_path)
        sys.argv = [script_path] + passthrough
        runpy.run_path(script_path, run_name="__main__")
    else:
        # Treat as a module name
        module = target
        sys.argv = [module] + passthrough
        runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
