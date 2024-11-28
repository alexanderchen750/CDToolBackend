def load_ebnf(file_path):
    """
    Loads the content of an ebnf file into a string with proper formatting.
    
    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content.strip()  # Removes any leading or trailing whitespace
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""