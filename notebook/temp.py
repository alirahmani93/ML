class A:
    def __init__(self,age=1):
        self._age:int = age
    @property
    def age(self):
        return self._age