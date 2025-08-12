#!/usr/bin/env python3
import sys
import os
import tiktoken

# Default encoding for modern models (like GPT-3.5, GPT-4, and a good proxy for Gemini)
DEFAULT_ENCODING = "cl100k_base"
# Fallback encoding if the default is somehow not available
FALLBACK_ENCODING = "gpt2"


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Counts tokens using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        # This fallback is unlikely to be needed with standard tiktoken installations
        # but good practice to have one.
        print(
            f"Warning: Encoding '{encoding_name}' not found. Falling back to '{FALLBACK_ENCODING}'.",
            file=sys.stderr,
        )
        encoding = tiktoken.get_encoding(FALLBACK_ENCODING)
    return len(encoding.encode(text))


if __name__ == "__main__":
    input_text = ""
    script_name = os.path.basename(sys.argv[0])  # Get script name for usage message

    if len(sys.argv) > 1:
        # Arguments are provided
        first_arg = sys.argv[1]
        if os.path.isfile(first_arg):
            # First argument is a file
            try:
                with open(first_arg, "r", encoding="utf-8") as f:
                    input_text = f.read()
                if len(sys.argv) > 2:
                    # Warn if there are more arguments after the filename
                    extra_args = " ".join(sys.argv[2:])
                    print(
                        f"Warning: Reading from file '{first_arg}'. Additional arguments ('{extra_args}') ignored.",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"Error: Could not read file '{first_arg}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # First argument is not a file, treat all arguments as a single string
            input_text = " ".join(sys.argv[1:])
    else:
        # No arguments, try to read from stdin
        if sys.stdin.isatty():
            # stdin is a TTY (interactive terminal) and no arguments were given
            print("Usage:", file=sys.stderr)
            print(
                f'  1. Provide text as argument: {script_name} "your text here"',
                file=sys.stderr,
            )
            print(
                f"  2. Provide a file path:      {script_name} path/to/yourfile.txt",
                file=sys.stderr,
            )
            print(
                f'  3. Pipe input:               echo "your text" | {script_name}',
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            # stdin is piped
            input_text = sys.stdin.read()

    # Perform token counting
    # Note: An empty input_text (e.g., from an empty file or empty pipe) is valid
    # and will typically result in 0 tokens.
    try:
        token_count = count_tokens(input_text)
        print(token_count)
    except Exception as e:
        print(f"Error during tokenization: {e}", file=sys.stderr)
        sys.exit(1)
