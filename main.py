from agent import run_pipeline


def main():
    result = run_pipeline("example.fits", modality="spectral")
    print(result)


if __name__ == "__main__":
    main()
