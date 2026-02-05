"""Run background estimation simulation."""

from nima.generat import run_simulation

if __name__ == "__main__":
    df = run_simulation(num_repeats=5)
    print(df.describe())
