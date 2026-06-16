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


class Matrix:
    def __init__(self,rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows),len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other,Vector):
            result = []
            for i in range(self.shape[0]):
                value = sum(a * b for a,b in zip(self.rows[i],other.vector))
                result.append(value)

            return Vector(result)
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                value = sum(self.rows[i][k] * other.rows[k][j] for k in range(self.shape[1]))
                row.append(value)

            rows.append(row)

        return Matrix(rows)

    def transpose(self):
        new_rows = []
        for i in range(self.shape[1]):
            new_row = []
            for j in range(self.shape[0]):
                new_row.append(self.rows[j][i])
            new_rows.append(new_row)
        return Matrix(new_rows)

rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])
rotated = rotation_90 @ point
print(f"Original: {point.vector}")
print(f"Rotated 90°: {rotated.vector}")
M = Matrix([[1, 2, 3], [4, 5, 6]])
print(f"Original shape: {M.shape}")
print(f"Transposed shape: {M.transpose().shape}")
print(f"Transposed: {M.transpose().rows}")

