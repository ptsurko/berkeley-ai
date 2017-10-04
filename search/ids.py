from dls import dls_search_internal


def ids_search(problem):
    cutoff = 1
    while True:
        result, cutoff_occurred = dls_search_internal(problem, cutoff)
        if not cutoff_occurred or result is not None:
            return result

        cutoff += 1

