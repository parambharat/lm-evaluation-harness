import os

import cohere
from lm_eval.base import BaseLM
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_results(client, context, response):
    tokenized = client.tokenize(context)
    ctxlen = len(tokenized.tokens) - 1
    continuation_probabilites = sum(
        token.likelihood
        for token in response.generations[0].token_likelihoods[ctxlen:]
        if token.likelihood is not None
    )
    return continuation_probabilites


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def cohere_generation(client, **kwargs):
    """Query cohere API for generation.

    Retry with back-off until they respond
    """
    try:
        return client.generate(
            model=kwargs.get("model", "small",),
            prompt=kwargs.get("prompt"),
            max_tokens=kwargs.get("max_tokens", 0),
            temperature=kwargs.get("temperature", 0),
            return_likelihoods=kwargs.get("return_likelihoods", "ALL"),
        )
    except cohere.CohereError as e:
        raise e


class CohereLM(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(
        self, engine,
    ):
        """

        :param model: str
            Cohere Model Name: eg small, medium, large, xlarge
        """
        super().__init__()
        import cohere

        self.engine = engine
        self.api_key = os.environ["COHERE_API_SECRET_KEY"]
        self.cohere_client = cohere.Client(self.api_key)

    @property
    def eot_token_id(self):
        raise NotImplementedError()

    @property
    def max_length(self):
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        raise NotImplementedError()

    def tok_decode(self, tokens):
        raise NotImplementedError()

    def loglikelihood(self, requests):
        res = []
        for context, continuation in tqdm(requests):
            response = cohere_generation(
                self.cohere_client,
                model=self.engine,
                prompt=f"{context} {continuation}",
                max_tokens=0,
                temperature=0.0,
                return_likelihoods="ALL",
            )
            logprob = get_results(self.cohere_client, context, response)
            self.cache_hook.add_partial("loglikelihood", context, logprob)
            res.append((logprob, True))
        return res

    def loglikelihood_rolling(self, requests):
        # TODO: The Cohere API does not support tokenized inputs so we cannot
        # manually partition long contexts into smaller rolling windows as
        # done for other models derived from `BaseLM`. Override this method
        # with a windowing scheme that works for direct string inputs.
        raise NotImplementedError(
            "`loglikelihood_rolling` is currently not supported due to lack of "
            "input tokenization support from Cohere."
        )

    def greedy_until(self, requests):
        if not requests:
            return []

        res = []
        for request in tqdm(requests):
            inp = request[0]
            until = request[1]
            response = cohere_generation(
                self.cohere_client,
                model=self.engine,
                prompt=inp,
                max_tokens=self.max_gen_toks,
                temperature=0.0,
                stop_sequences=[until],
            )
            resp = response.generations[0].text
            self.cache_hook.add_partial("greedy_until", (inp, until), resp)
            res.append(resp)
        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
