def time_display(s):
    d = s // (3600*24)
    s -= d * (3600*24)
    h = s // 3600
    s -= h * 3600
    m = s // 60
    s -= m * 60
    str_time = "{:1d}d ".format(int(d)) if d else " "
    return str_time + "{:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s))
