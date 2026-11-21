"""Metal constants for expanding a byte containing two native E2M1 weights."""

def table():
    levels = ("0.", ".5", "1.", "1.5", "2.", "3.", "4.", "6.",
              "-0.", "-.5", "-1.", "-1.5", "-2.", "-3.", "-4.", "-6.")
    entries = [f"half2({levels[i & 15]}h,{levels[i >> 4]}h)" for i in range(256)]
    return "constant half2 nibble_pairs[256]={" + ",".join(entries) + "};"
