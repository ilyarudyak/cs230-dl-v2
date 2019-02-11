REFERENCE1 = """The cat is on the mat"""

REFERENCE2 = """There is a cat on the mat"""

CANDIDATE1 = """the the the the the the the"""


class Example2:
    def __init__(self):
        self.reference1 = self._get_list(REFERENCE1)
        self.reference2 = self._get_list(REFERENCE2)
        self.candidate1 = self._get_list(CANDIDATE1)
        self.reference_list = [self.reference1, self.reference2]

    @staticmethod
    def _get_list(text):
        return text.lower().split()
