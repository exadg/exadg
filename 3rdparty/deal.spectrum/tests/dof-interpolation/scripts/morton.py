
class Morton(object):

    def __init__(self, dimensions=2, bits=32):
        assert dimensions > 0, 'dimensions should be greater than zero'
        assert bits > 0, 'bits should be greater than zero'

        def flp2(x):
            '''Greatest power of 2 less than or equal to x, branch-free.'''
            x |= x >> 1
            x |= x >> 2
            x |= x >> 4
            x |= x >> 8
            x |= x >> 16
            x |= x >> 32
            x -= x >> 1
            return x

        shift = flp2(dimensions * (bits - 1))

        masks = []
        lshifts = []
        max_value = (1 << (shift*bits))-1
        while shift > 0:
            mask = 0
            shifted = 0
            for bit in range(bits):
                distance = (dimensions * bit) - bit
                shifted |= shift & distance
                mask |= 1 << bit << (((shift - 1) ^ max_value) & distance)

            if shifted != 0:
                masks.append(mask)
                lshifts.append(shift)

            shift >>= 1

        self.dimensions = dimensions
        self.bits = bits
        self.lshifts = [0] + lshifts
        self.rshifts = lshifts + [0]
        self.masks = [(1 << bits) - 1] + masks

    def __repr__(self):
        return '<Morton dimensions={}, bits={}>'.format(
            self.dimensions, self.bits)

    def split(self, value):
        # type: (int) -> int
        masks = self.masks
        lshifts = self.lshifts
        for o in range(len(masks)):
            value = (value | (value << lshifts[o])) & masks[o]
        return value

    def compact(self, code):
        # type: (int) -> int
        masks = self.masks
        rshifts = self.rshifts
        for o in range(len(masks)-1, -1, -1):
            code = (code | (code >> rshifts[o])) & masks[o]
        return code

    def shift_sign(self, value):
        # type: (int) -> int
        assert not(value >= (1<<(self.bits-1)) or value <= -(1<<(self.bits-1))), (value, self.bits)
        if value < 0:
            value = -value
            value |= 1 << (self.bits - 1)
        return value

    def unshift_sign(self, value):
        # type: (int) -> int
        sign = value & (1 << (self.bits - 1))
        value &= (1 << (self.bits - 1)) - 1
        if sign != 0:
            value = -value
        return value

    def pack(self, *args):
        # type: (List[int]) -> int
        assert len(args) <= self.dimensions
        assert all([v < (1 << self.bits) for v in args])

        code = 0
        for i in range(self.dimensions):
            code |= self.split(args[i]) << i
        return code

    def unpack(self, code):
        # type: (int) -> List[int]
        values = []
        for i in range(self.dimensions):
            values.append(self.compact(code >> i))
        return values

    def spack(self, *args):
        # type: (List[int]) -> int
        return self.pack(*map(self.shift_sign, args))

    def sunpack(self, code):
        # type: (int) -> List[int]
        values = self.unpack(code)
        return list(map(self.unshift_sign, values))

    def __eq__(self, other):
        return (
            self.dimensions == other.dimensions and
            self.bits == other.bits
        )