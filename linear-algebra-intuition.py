import math

class Vector():
    def __init__(self,components):
        self.components = list(components)
        self.dimension = len(components)

    def __repr__(self):
        return f"Vector({self.components})"

    def __str__(self):
        return f"Vector({self.components})"

    def __len__(self):
        return self.dimension

    def __getitem__(self, item):
        return self.components[item]

    def __eq__(self, other):
        return (self.dimension == other.dimension and
                all(abs(a - b) < 1e-9 for a, b in zip(self.components, other.components)))

    def _check_same_dim(self, other):
        if self.dimension != other.dimension:
            raise ValueError(f"dim mismatch: {self.dimension} vs {other.dimension}")

    def __add__(self, other):
        self._check_same_dim(other)
        return Vector([a+b for a,b in zip(self.components,other.components)])

    def __sub__(self, other):
        self._check_same_dim(other)
        return Vector([a-b for a,b in zip(self.components,other.components)])

    def __mul__(self, scaler):
        return Vector([a*scaler for a in self.components])

    def __rmul__(self, scaler):
        return self * scaler

    def __truediv__(self, scaler):
        return Vector([a/scaler for a in self.components])

    def __neg__(self):
        return Vector([a*-1 for a in self.components])

    def dot(self,other):
        self._check_same_dim(other)
        return sum(a * b for a,b in zip(self.components,other.components))

    def magnitude(self):
        return self.dot(self) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("cannot normalize the zero vector")
        return self / mag

    def cosine_similarity(self,other):
        return self.dot(other)/(self.magnitude() * other.magnitude())

    def angle_between(self, other):
        cos = self.cosine_similarity(other)
        cos = max(-1.0, min(1.0, cos))
        radians = math.acos(cos)
        return math.degrees(radians)

    def distance(self,other):
        res = self - other
        distance = res.magnitude()
        return distance

    def projection(self,other):
        scalar = self.dot(other)/other.dot(other)
        return scalar * other



