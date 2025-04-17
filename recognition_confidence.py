class _DigitScore:

	def __init__(self, digit:int, score:int=0):
		self._digit = digit
		self._score = score

	@property
	def digit(self) -> int:
		return self._digit

	@property
	def score(self) -> int:
		return self._score

	@score.setter
	def score(self, value:int) -> None:
		self._score = value

	def apply_delta_score(self, delta_score:float) -> None:
		self._score += delta_score


class RecognitionConfidence:

	def __init__(self) -> None:
		self._content = dict()

	def __getitem__(self, digit_key:int) -> int:
		return self._get_digit_score(digit_key).score

	def __setitem__(self, digit_key:int, score:int) -> None:
		self._get_digit_score(digit_key).score = score

	def apply_delta_score(self, digit_key:int, delta_score:int) -> None:
		self._get_digit_score(digit_key).apply_delta_score(delta_score)

	def as_dict(self) -> dict[int, float]:
		return {digit: ds.score for digit, ds in self._content.items()}

	def calculate_confidence(self) -> float:
		sorted_scores = sorted(
			self._content.values(), key=lambda ds: ds.score, reverse=True)
		max_ds0 = sorted_scores[0]
		max_ds1 = sorted_scores[1]

		confidence = min((1.0 / max_ds0.score) * max_ds1.score, 1.0)
		return max_ds0.digit, confidence

	def _get_digit_score(self, digit_key:int) -> _DigitScore:
		ds = self._content.get(digit_key)

		if ds is None:
			ds = _DigitScore(digit_key)
			self._content[digit_key] = ds

		return ds
