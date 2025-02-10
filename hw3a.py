from numericalMethods import GPDF, Probability, Secant


def find_c(mean, stDev, target_prob, GT, OneSided):
    """
    Uses the Secant method to find c for a given probability.
    """

    def error_function(c):
        if OneSided:
            return Probability(GPDF, (mean, stDev), c, GT) - target_prob
        else:
            return (1 - 2 * Probability(GPDF, (mean, stDev), c, GT=True)) - target_prob

    c_guess1, c_guess2 = mean, mean + stDev  # Initial guesses
    c_solution, _ = Secant(error_function, c_guess1, c_guess2)
    return c_solution


def main():
    Again = True
    yesOptions = ["y", "yes", "true"]

    while Again:
        mean = float(input("Population mean? (default=0):") or 0)
        stDev = float(input("Standard deviation? (default=1):") or 1)
        mode = input("Are you specifying c (to find P) or P (to find c)? Enter 'c' or 'p': ").strip().lower()

        OneSided = input("One-sided probability? (default=True) (Y/N):").strip().lower() in yesOptions
        GT = input("Probability greater than c? (default=False) (Y/N):").strip().lower() in yesOptions

        if mode == 'c':
            c = float(input("Enter c: "))
            if OneSided:
                prob = Probability(GPDF, (mean, stDev), c, GT)
                print(f"P(x {'>' if GT else '<'} {c:.2f} | {mean:.2f}, {stDev:.2f}) = {prob:.3f}")
            else:
                prob = 1 - 2 * Probability(GPDF, (mean, stDev), c, GT=True)
                print(
                    f"P({mean - (c - mean):.2f} < x < {mean + (c - mean):.2f} | {mean:.2f}, {stDev:.2f}) = {prob:.3f}")

        elif mode == 'p':
            target_prob = float(input("Enter probability P: "))
            c_value = find_c(mean, stDev, target_prob, GT, OneSided)
            print(f"Value of c for probability {target_prob:.3f}: c = {c_value:.3f}")

        Again = input("Go again? (Y/N):").strip().lower() in yesOptions


if __name__ == "__main__":
    main()