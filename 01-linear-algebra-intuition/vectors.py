class Vector:
    def __init__(self,vector):
        self.vector = vector

    def __add__(self, other):
        return Vector([a + b for a,b in zip(self.vector,other.vector)])

    def __sub__(self, other):
        return Vector([a - b for a,b in zip(self.vector,other.vector)])

    def dot(self,other):
        return sum(a * b for a,b in zip(self.vector,other.vector))

    def magnitude(self):
        return sum([a**2 for a in self.vector]) ** 0.5

    def normalize(self):
        magnitude = self.magnitude()
        return Vector([a / magnitude for a in self.vector])

    def cosine_similarity(self,other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {(a + b).vector}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")