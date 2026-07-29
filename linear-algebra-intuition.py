import math


class Vector():
    """A dense vector with from-scratch geometric operations (no NumPy).

    Supports arithmetic, dot product, magnitude, normalization, cosine
    similarity, angle, distance, and projection. Immutable: every
    operation returns a new Vector.
    """
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

class Matrix():
    def __init__(self,mat):
        self.mat = [list(row) for row in mat]
        self.shape = (len(mat),len(mat[0]))

    def __repr__(self):
        return f"Matrix({self.mat})"

    def __str__(self):
        return f"Matrix({self.mat})"

    def transpose(self):
        res = []
        for i in range(self.shape[1]):
            new_row = []
            for j in range(self.shape[0]):
                new_row.append(self.mat[j][i])
            res.append(new_row)

        return Matrix(res)

    def __matmul__(self, other):
        if isinstance(other, Vector):
            if self.shape[1] != other.dimension:
                raise ValueError(f"dim mismatch: {self.shape} @ {other.dimension}")
            return Vector([
                sum(self.mat[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])

        if self.shape[1] != other.shape[0]:
            raise ValueError(f"inner dims must match: {self.shape} @ {other.shape}")

        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                s = 0
                for k in range(self.shape[1]):
                    s += self.mat[i][k] * other.mat[k][j]
                row.append(s)
            rows.append(row)
        return Matrix(rows)

def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    rows = [v.components[:] for v in vectors]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n

def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            w = w - w.projection(u)
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal
