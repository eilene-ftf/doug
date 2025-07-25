import torchhd

from encode.encode import EncodingEnvironment
from language.syntax import LLBool


def main():
    enc_env = EncodingEnvironment(dim=100)
    # print(list(enc_env.codebook.items()))
    b = LLBool(1)
    benc = enc_env.encode_type(b)

    print(
        enc_env.codebook["bool"].cosine_similarity(
            benc.bind(enc_env.codebook["#:kind"].inverse())
        )
    )


if __name__ == "__main__":
    main()
