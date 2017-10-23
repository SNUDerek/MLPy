from math import sqrt, acos, pi

#####

# VECTOR CLASS

#####

class Vector(object):

    CANNOT_NORMALIZE_ZERO_VECTOR = 'cannot normalize zero vector'
    VECTOR_LENGTHS_NOT_EQUAL = 'vector lengths not equal'
    NO_UNIQUE_PARALLEL_COMPONENT = 'no unique parallel component!'

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('coordinates must be non-empty')

        except TypeError:
            raise TypeError('coordinates must be an iterable')

        def __str__(self):
            return 'vector: {}'.format(self.coordinates)

        def __eq__(self, v):
            return self.coordinates == v.coordinates

    # vector addition
    def plus(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            result = [x+y for x, y in zip(self.coordinates, vec.coordinates)]

            return(Vector(result))

        except ValueError:
            raise ValueError('vector lengths not equal')

    # vector subtraction
    def minus(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            result = [x-y for x, y in zip(self.coordinates, vec.coordinates)]

            return(Vector(result))

        except ValueError:
            raise ValueError('vector lengths not equal')

    # scalar multiplication
    def scalarmultiply(self, scalar):
        result = [x*scalar for x in self.coordinates]
        return(Vector(result))
        
    def scalarmult(self, scalar):
        return(self.scalarmultiply(scalar))

    # vector magnitude: square root of sum of squares
    def magnitude(self):
        denom = [x**2 for x in self.coordinates]
        result = sqrt(sum(denom))
        return(result)

    # normalized vector (inputs considered points)
    def normalized(self):
        try:
            denom = self.magnitude()
            result = [x/denom for x in self.coordinates]
            return(Vector(result))
            
        except ZeroDivisionError:
            raise Exception('cannot normalize zero vector')

    # direction unit vector (as values)
    def direction(self):
        dvec = self.normalized()
        return(dvec.coordinates)

    # vector inner/dot product (sum of piecewise multiplication)
    def inner(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            temp = [x*y for x, y in zip(self.coordinates, vec.coordinates)]
            return(sum(temp))

        except ValueError:
            raise ValueError('vector lengths not equal')

    def dot(self, vec):
        return(self.inner(vec))

    # angle between two vectors as cosine
    def cosineto(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            mags = self.magnitude() * vec.magnitude()
            if mags == 0:
                print('notice: attempting to find cosine of 0 vector')
                return(0.0)
            else:
                return(self.dot(vec) / mags)

        except ValueError:
            raise ValueError('vector lengths not equal')
        
    # angle between two vectors as radians
    # a dot b = mag a * mag b * cos theta
    def radiansto(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            rads = acos(self.cosineto(vec))
            return(rads)

        except ValueError:
            raise ValueError('vector lengths not equal')
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR:
                raise Exception('cannot compute angle with zero vector')

    def radsto(self, vec):
        return(self.radiansto(vec))
        
    # angle between two vectors as degrees
    # degrees = rads * 180 / pi
    def degreesto(self, vec):
        return(self.radiansto(vec)*180/pi)
    def degsto(self, vec):
        return(self.degreesto(vec))

    # check for parallel (cos = -1, 1)
    def isparallel(self, vec):
        if vec.magnitude() == 0 or self.magnitude() == 0:
            return(True)
        cos = self.cosineto(vec)
        if (0.999 < cos < 1.001) or (-1.001 < cos < -0.999):
            return(True)
        else:
            return(False)

    # check for orthogonal (cos = 0)
    def isorthogonal(self, vec):
        cos = self.cosineto(vec)
        if (-0.001 < cos < 0.001):
            return(True)
        else:
            return(False)

    def isright(self, vec):
        return(self.isorthogonal(vec))

    # projection of v to b: proj_b(v) = (v dot unit_b) * unit_b
    def projection(self, vec):
        try:
            unit_vec = vec.normalized()
            return(unit_vec.scalarmult(self.dot(unit_vec)))

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR:
                raise Exception('no unique parallel component!')
            else:
                raise e

    def componentparallelto(self, vec):
        return(self.projection(vec))

    # orthogonal of v from b: v minus proj_b(v)
    def orthogonal(self, vec):
        try:
            proj_vec = self.projection(vec)
            return(self.minus(proj_vec))

        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT:
                raise Exception('no unique orthogonal component!')
            else:
                raise e

    def componentorthogonalto(self, vec):
        return(self.orthogonal(vec))

    # cross product: v x w orthogonal to v and w
    #                ||v x w|| = ||v|| ||w|| sin theta
    #                direction found by 'right hand rule'
    #                'anticommutative': order matters
    def cross(self, vec):
        try:
            if not (1 < len(self.coordinates) < 4) or not (1 < len(
                self.coordinates) < 4):
                raise ValueError
            elif len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            v1 = list(self.coordinates)
            v2 = list(vec.coordinates)

            if len(self.coordinates) == 2:
                v1.insert(0, 0)
                v2.insert(0, 0)
            
            a = v1[1]*v2[2] - v2[1]*v1[2]
            b = v1[0]*v2[2] - v2[0]*v1[2]
            c = v1[0]*v2[1] - v2[0]*v1[1]
            result = [a, -b, c]
            return(Vector(result))

        except ValueError:
            raise ValueError('vector lengths must be equal; dim 2 or 3')

    # area of triangle spanned by two vectors = 1/2 mag (a cross b)
    def trianglearea(self, vec):
        rvec = self.cross(vec)
        return(0.5*rvec.magnitude())
    
#####

# LINE CLASS

#####

# if separating classes by file...
# from vector import Vector

class Line(object):

    NON_NONZERO_ELMTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 2

        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

        if not constant_term:
            constant_term = 0.0
        self.constant_term = constant_term