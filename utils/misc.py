def in_google_colab():
    """Checks if the code is running in Google Colab

    Returns:
        bool: If True, the code is running in Google Colab environment
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False