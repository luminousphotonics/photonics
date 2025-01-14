import subprocess

repo = "/opt/homebrew/Library/Taps/homebrew/homebrew-core"  # Path to the Homebrew core tap

def get_python2_formula_at_commit(commit_hash):
    """Retrieves the python@2.rb formula from a specific commit."""
    try:
        context = subprocess.check_output(
            ['git', '-C', repo, 'show', f'{commit_hash}:Formula/python@2.rb']
        ).decode()
        return context
    except subprocess.CalledProcessError:
        return None

def find_last_python2_commit():
    """Finds the last commit that modified python@2.rb."""
    try:
        rev_list = subprocess.check_output(
            ['git', '-C', repo, 'rev-list', 'HEAD', '--', 'Formula/python@2.rb']
        ).strip().decode().splitlines()
        
        if rev_list:
            return rev_list[0]  # Return the first (most recent) commit
        else:
            return None
    except subprocess.CalledProcessError:
        return None

# Find the last commit
last_commit = find_last_python2_commit()

if last_commit:
    print(f"Last commit with python@2.rb: {last_commit}")

    # Get the formula content (optional - you might just want the commit hash)
    formula_content = get_python2_formula_at_commit(last_commit)
    if formula_content:
        print("Content of python@2.rb at that commit:")
        print(formula_content)

    # Install python@2 from the commit
    print("Installing python@2 from the commit...")
    subprocess.run(['brew', 'install', f'https://raw.githubusercontent.com/Homebrew/homebrew-core/{last_commit}/Formula/python@2.rb'])

else:
    print("Could not find a commit with python@2.rb in the Homebrew repository.")