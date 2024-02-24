def chunk_list(input_list, chunk_size):
    """Yield successive chunk_size chunks from input_list."""
    for i in range(0, len(input_list), chunk_size):
        yield i, input_list[i:i + chunk_size]