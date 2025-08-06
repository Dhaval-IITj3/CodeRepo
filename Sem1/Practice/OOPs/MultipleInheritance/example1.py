class Animal():
    def __init__(self, name):
        print("Animal created")
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement this abstract method")


class Mamal(Animal):
    def __init__(self, name):
        print("Mamal created")
        super().__init__(name)

    def speak(self):
        print(f"{self.name} says hello!")


class Amphibian(Animal):
    def __init__(self, name):
        print("Amphibian created")
        super().__init__(name)

    def speak(self):
        print(f"{self.name} says hello!")


class Domestic():
    def __init__(self, name):
        print("Domestic created")
        self.name = name

    def intro(self):
        print(f"I, {self.name}, am a domestic animal.")


class Wild():
    def __init__(self, name):
        print("Wild created")
        self.name = name

    def intro(self):
        print(f"Beware!!! I, {self.name}, am a wild animal.")


class Dog(Mamal, Domestic):
    def __init__(self, name):
        print("Dog created")
        super().__init__(name)
        Domestic.__init__(self, name)

    def intro(self):
        print(f"{self.name} woofs")


if __name__ == "__main__":
    mydog = Dog("Tommy")
    mydog.speak()
    mydog.intro()
