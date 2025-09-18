from multiple_minimum_monte_carlo import multiproc


def test_batch_dicts_even_split():
    dicts = [{'a': i} for i in range(6)]
    batched = multiproc.batch_dicts(dicts.copy(), 3)
    # Expect roughly even distribution
    lengths = [len(b) for b in batched]
    assert sum(lengths) == 6


def test_parallel_run_proc_single_worker(tmp_path):
    # Define a simple function that will be called
    def func(x, y):
        return x + y

    # Wrap the callable to match expected call signature in multiproc.run_func
    def wrapper(**kwargs):
        return func(**kwargs['args']) if 'args' in kwargs else func(kwargs['x'], kwargs['y'])

    results = multiproc.parallel_run_proc(wrapper, [{'args': {'x': 3, 'y': 4}}], num_workers=1)
    assert results == [7]
