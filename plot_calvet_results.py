"""Plot persisted results without raw data or onset calculations."""
import argparse
from pathlib import Path
from util.calvet_plotting import compare_saved, plot_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="results.json saved by process_dsc_data.py")
    parser.add_argument("--plot", choices=["signals", "pt", "hf-time", "hf-temperature", "compare"],
                        type=lambda value: "signals" if value == "diagnostic" else value, default="signals")
    parser.add_argument("--event-id")
    parser.add_argument("--out", default="out/figures/calvet_comparison.png")
    parser.add_argument("--x", choices=["T_muestra", "t_h"], default="T_muestra")
    args = parser.parse_args()
    if args.plot == "compare":
        print(compare_saved(args.results, Path(args.out), args.x))
    else:
        for path in args.results:
            print(plot_saved(path, args.plot, args.event_id))


if __name__ == "__main__":
    main()
