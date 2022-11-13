# This is an implementation of JSD loss as described in the SpanPredict paper
import torch

# TODO: separate checks in test() into a pytest file


def H(p):
    # p has shape  B, S,1 or B,S, J
    # do this along axis 1
    # -sum(p*log(p))
    return -torch.sum(p*torch.log(p + 1e-40), 1)  # B,1 or B,J


def JSD(r, theta=0.5):
    # r has shape B, S, J
    # JSD = H(sum(pi_j*r_j)) - H(pi_j*sum(r_j) for j = 1 .. J with mixing coeffs pi_j
    # we'll be using even mixing, so that's
    # JSD = H(mean(r_j)) - H(mean(sum(r_j))
    # Can be interpreted as  JSD = span_overlap - span_conciseness
    # bounded by [0, log(J)]
    # can be weighted like:
    # 2*( theta*span_overlap - (1-theta)*span_conciseness)
    # for theta on [0, 0.5], but no longer bounded below at 0, can be negative

    J = r.shape[2]
    overlap = H(torch.mean(r, dim=2).unsqueeze(2)).squeeze(1)  # B,1
    conciseness = torch.mean(H(r), dim=1)

    jsd = 2.0*(theta*overlap - (1-theta)*conciseness)  # TODO: tensors, to(device)
    return jsd


def test():

    inputs = [
        [
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        ],
        [
            [
                [0.5, 0.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [0.0, 0.5],
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        ],
        [
            [
                [0.5, 0.0],
                [0.5, 0.0],
                [0.0, 0.25],
                [0.0, 0.25],
                [0.0, 0.25],
                [0.0, 0.25]
            ]
        ],
        [
            [
                [0.5, 0.0],
                [0.5, 0.5],
                [0.0, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        ],
        [
            [
                [0.5, 0.0],
                [0.5, 0.25],
                [0.0, 0.25],
                [0.0, 0.25],
                [0.0, 0.25],
                [0.0, 0.0]
            ]
        ]
    ]

    for r in inputs:
        r = torch.tensor(r)
        jsd = JSD(r, theta=0.5)
        print(jsd)

    for r in inputs:
        r = torch.tensor(r)
        jsd = JSD(r, theta=0.45)
        print(jsd)


if __name__ == "__main__":
    test()
