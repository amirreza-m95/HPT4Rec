import torch


def model():
    m = torch.nn.Linear(40, 100)
    return m


def main():
    m = model().to("cuda")
    data = torch.rand(size=(20000, 40), device="cuda")
    m.train()
    for _ in range(100000):
        out = m(data)
        print(out)


if __name__ == "__main__":
    main()
