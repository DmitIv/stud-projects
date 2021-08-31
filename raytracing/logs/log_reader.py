import pstats


def log2text(log_file: str):
    s = pstats.Stats(log_file).sort_stats("tottime")
    s.print_stats()
    return s.stats


if __name__ == "__main__":
    log = log2text("rendering_per_pixel.prof")
