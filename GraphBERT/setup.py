#!/usr/bin/env python3
"""
install_and_check_dependencies.py

Installs and verifies all dependencies for:
- data.py (C++ DFG extraction)
- model.py (Model definitions)
- train.py (C++ training)
- evaluate.py (Model evaluation)

Usage:
    python install_and_check_dependencies.py [--install] [--upgrade]

Options:
    --install    Install missing packages
    --upgrade    Upgrade all packages to latest versions
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


# Core dependencies required by all scripts
CORE_DEPENDENCIES = {
    'torch': 'torch>=2.0.0',
    'transformers': 'transformers>=4.30.0',
    'numpy': 'numpy>=1.21.0',
    'tqdm': 'tqdm>=4.60.0',
}

# Data processing dependencies (for data.py)
DATA_DEPENDENCIES = {
    'datasets': 'datasets>=2.10.0',
    'tree_sitter': 'tree-sitter>=0.20.0',
}

# C++ tree-sitter binding
CPP_TREE_SITTER = {
    'tree_sitter_cpp': 'tree-sitter-cpp>=0.20.0',
}

# Training dependencies (for train.py)
TRAIN_DEPENDENCIES = {
    'torch.cuda.amp': 'torch',  # Part of torch
}

# Evaluation dependencies (for evaluate.py)
EVAL_DEPENDENCIES = {
    'tree_sitter': 'tree-sitter>=0.20.0',
    'tree_sitter_cpp': 'tree-sitter-cpp>=0.20.0',
}

# Organize by category
DEPENDENCIES = {
    'Core (Required by all)': CORE_DEPENDENCIES,
    'Data Processing (data.py)': {**DATA_DEPENDENCIES, **CPP_TREE_SITTER},
    'Training (train.py)': {},  # Uses core packages
    'Evaluation (evaluate.py)': {**EVAL_DEPENDENCIES},
}


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and get version.
    
    Args:
        package_name: Name of the package (as installed)
        import_name: Python import name (if different from package_name)
        
    Returns:
        Tuple of (is_installed, version_string)
    """
    import_name = import_name or package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, 'not installed'


def install_package(package_spec: str) -> bool:
    """
    Install a package using pip.
    
    Args:
        package_spec: Package specification (e.g., 'torch>=2.0.0')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package_spec],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def upgrade_package(package_spec: str) -> bool:
    """
    Upgrade a package using pip.
    
    Args:
        package_spec: Package specification
        
    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', package_spec],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def check_all_dependencies() -> Dict[str, Dict[str, Tuple[bool, str]]]:
    """
    Check all dependencies across all categories.
    
    Returns:
        Dictionary mapping category → {package: (installed, version)}
    """
    results = {}
    
    for category, packages in DEPENDENCIES.items():
        results[category] = {}
        for import_name, package_spec in packages.items():
            is_installed, version = check_package(import_name)
            results[category][import_name] = (is_installed, version)
    
    return results


def print_dependency_report(results: Dict[str, Dict[str, Tuple[bool, str]]]):
    """Print a formatted report of dependency status."""
    print_header("Dependency Status Report")
    
    total_packages = 0
    installed_packages = 0
    
    for category, packages in results.items():
        print(f"{Colors.BOLD}{category}{Colors.RESET}")
        print("-" * 80)
        
        for package_name, (is_installed, version) in packages.items():
            total_packages += 1
            if is_installed:
                installed_packages += 1
                status = f"{Colors.GREEN}✓ INSTALLED{Colors.RESET}"
                print(f"  {package_name:30} {status:25} ({version})")
            else:
                status = f"{Colors.RED}✗ MISSING{Colors.RESET}"
                print(f"  {package_name:30} {status:25}")
        
        print()
    
    # Summary
    print(f"{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  Total packages:      {total_packages}")
    print(f"  Installed:           {installed_packages}")
    print(f"  Missing:             {total_packages - installed_packages}")
    print()
    
    if installed_packages == total_packages:
        print_success("All dependencies installed!")
    else:
        print_warning(f"{total_packages - installed_packages} package(s) missing")


def install_missing(results: Dict[str, Dict[str, Tuple[bool, str]]]) -> bool:
    """
    Install all missing packages.
    
    Returns:
        True if all installations succeeded, False otherwise
    """
    print_header("Installing Missing Packages")
    
    missing_packages = []
    package_specs = []
    
    for category, packages in DEPENDENCIES.items():
        for import_name, package_spec in packages.items():
            if not results[category][import_name][0]:
                missing_packages.append(import_name)
                package_specs.append(package_spec)
    
    if not missing_packages:
        print_success("No missing packages!")
        return True
    
    print(f"Found {len(missing_packages)} missing package(s):\n")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    print()
    
    all_success = True
    for package_spec in package_specs:
        print(f"Installing {package_spec}...", end=" ", flush=True)
        if install_package(package_spec):
            print_success("done")
        else:
            print_error("failed")
            all_success = False
    
    return all_success


def upgrade_all(results: Dict[str, Dict[str, Tuple[bool, str]]]) -> bool:
    """
    Upgrade all installed packages.
    
    Returns:
        True if all upgrades succeeded, False otherwise
    """
    print_header("Upgrading All Packages")
    
    all_specs = []
    for category, packages in DEPENDENCIES.items():
        for import_name, package_spec in packages.items():
            if results[category][import_name][0]:  # Only upgrade installed
                all_specs.append(package_spec)
    
    if not all_specs:
        print_warning("No installed packages to upgrade")
        return True
    
    print(f"Upgrading {len(all_specs)} package(s)...\n")
    
    all_success = True
    for package_spec in all_specs:
        pkg_name = package_spec.split('>=')[0].split('==')[0]
        print(f"Upgrading {pkg_name}...", end=" ", flush=True)
        if upgrade_package(package_spec):
            print_success("done")
        else:
            print_error("failed")
            all_success = False
    
    return all_success


def test_imports() -> bool:
    """Test that critical imports work."""
    print_header("Testing Critical Imports")
    
    critical_imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('datasets', 'Datasets'),
        ('tree_sitter', 'Tree-sitter'),
    ]
    
    all_success = True
    
    for import_name, display_name in critical_imports:
        try:
            __import__(import_name)
            print_success(f"{display_name} imports correctly")
        except ImportError as e:
            print_error(f"{display_name} import failed: {e}")
            all_success = False
    
    # Special test for tree-sitter-cpp
    print("\nTesting tree-sitter C++ binding...", end=" ", flush=True)
    try:
        import tree_sitter_cpp
        print_success("C++ binding available")
    except ImportError:
        print_warning("C++ binding not available (optional, needed for data.py)")
    
    return all_success


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Install and check dependencies for GraphCodeBERT scripts'
    )
    parser.add_argument(
        '--install',
        action='store_true',
        help='Install missing packages'
    )
    parser.add_argument(
        '--upgrade',
        action='store_true',
        help='Upgrade all packages to latest versions'
    )
    
    args = parser.parse_args()
    
    # Step 1: Check all dependencies
    print_header("Checking Dependencies")
    results = check_all_dependencies()
    print_dependency_report(results)
    
    # Step 2: Install if requested
    if args.install:
        if not install_missing(results):
            print_error("Some installations failed. Check errors above.")
            return False
        
        # Re-check after installation
        results = check_all_dependencies()
        print_dependency_report(results)
    
    # Step 3: Upgrade if requested
    if args.upgrade:
        if not upgrade_all(results):
            print_error("Some upgrades failed. Check errors above.")
            return False
        
        # Re-check after upgrade
        results = check_all_dependencies()
        print_dependency_report(results)
    
    # Step 4: Test imports
    print()
    if not test_imports():
        print_warning("Some imports failed. You may need to install missing packages.")
        if not args.install:
            print_info("Run with --install flag to automatically install missing packages")
            return False
    
    # Final summary
    print_header("Setup Complete")
    
    missing_count = sum(
        1 for category in results.values()
        for is_installed, _ in category.values()
        if not is_installed
    )
    
    if missing_count == 0:
        print_success("All dependencies are installed and working!")
        print("\nYou can now run:")
        print("  python data.py --output_file data/cpp_functions.jsonl")
        print("  python train.py --data_file data/cpp_functions.jsonl --output_dir models --epochs 2")
        print("  python evaluate.py --model_checkpoint best_model")
        print()
        return True
    else:
        print_error(f"{missing_count} package(s) still missing")
        print_info("Run: python install_and_check_dependencies.py --install")
        print()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)