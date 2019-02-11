REFERENCE1 = """It is a guide to action that ensures that the military 
                will forever heed Party commands"""

REFERENCE2 = """It is the guiding principle which guarantees the military 
                forces always being under the command of the Party"""

REFERENCE3 = """It is the practical guide for the army always to heed 
                the directions of the party"""

CANDIDATE1 = """It is a guide to action which ensures that the military 
                always obeys the commands of the party"""

CANDIDATE2 = """It is to insure the troops forever hearing the activity 
                guidebook that party direct"""


class Example1:
    def __init__(self):
        self.reference1 = self._get_list(REFERENCE1)
        self.reference2 = self._get_list(REFERENCE2)
        self.reference3 = self._get_list(REFERENCE3)
        self.candidate1 = self._get_list(CANDIDATE1)
        self.candidate2 = self._get_list(CANDIDATE2)
        self.reference_list = [self.reference1, self.reference2, self.reference3]

    @staticmethod
    def _get_list(text):
        return text.lower().split()


