import numpy as np

from tomopy_cli import log


def theta_step(start, end, proj_number):
    return (end-start)/proj_number

def positive_int(value):
    """Convert *value* to an integer and make sure it is positive."""
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('Only positive integers are allowed')

    return result

def range_list(value):
    """
    Split *value* separated by ':' into int triple, filling missing values with 1s.
    """
    def check(region):
        if region[0] >= region[1]:
            raise argparse.ArgumentTypeError("{} must be less than {}".format(region[0], region[1]))

    lst = [int(x) for x in value.split(':')]

    if len(lst) == 1:
        frm = lst[0]
        return (frm, frm + 1, 1)

    if len(lst) == 2:
        check(lst)
        return (lst[0], lst[1], 1)

    if len(lst) == 3:
        check(lst)
        return (lst[0], lst[1], lst[2])

    raise argparse.ArgumentTypeError("Cannot parse {}".format(value))

def restricted_float(x):

    x = float(x)
    if x < 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def flip_and_stitch(params, img360, flat360, dark360):

    center = int(params.rotation_axis_flip)
    img = np.zeros([img360.shape[0]//2,img360.shape[1],2*img360.shape[2]-2*center],dtype=img360.dtype)
    flat = np.zeros([flat360.shape[0],flat360.shape[1],2*flat360.shape[2]-2*center],dtype=img360.dtype)
    dark = np.zeros([dark360.shape[0],dark360.shape[1],2*dark360.shape[2]-2*center],dtype=img360.dtype)
    img[:,:,img360.shape[2]-2*center:] = img360[:img360.shape[0]//2,:,:]
    img[:,:,:img360.shape[2]] = img360[img360.shape[0]//2:,:,::-1]
    flat[:,:,flat360.shape[2]-2*center:] = flat360
    flat[:,:,:flat360.shape[2]] = flat360[:,:,::-1]
    dark[:,:,dark360.shape[2]-2*center:] = dark360
    dark[:,:,:dark360.shape[2]] = dark360[:,:,::-1]

    params.rotation_axis = img.shape[2]//2

    return img, flat, dark


def patch_projection(data, miss_angles):

    fdatanew = np.fft.fft(data,axis=2)

    w = int((miss_angles[1]-miss_angles[0]) * 0.3)


    fdatanew[miss_angles[0]:miss_angles[0]+w,:,:] = np.fft.fft(data[miss_angles[0]-1,:,:],axis=1)
    fdatanew[miss_angles[0]:miss_angles[0]+w,:,:] *= np.reshape(np.cos(np.pi/2*np.linspace(0,1,w)),[w,1,1])

    fdatanew[miss_angles[1]-w:miss_angles[1],:,:] = np.fft.fft(data[miss_angles[1]+1,:,:],axis=1)
    fdatanew[miss_angles[1]-w:miss_angles[1],:,:] *= np.reshape(np.sin(np.pi/2*np.linspace(0,1,w)),[w,1,1])

    fdatanew[miss_angles[0]+w:miss_angles[1]-w,:,:] = 0
    # lib.warning("  *** %d, %d, %d " % (datanew.shape[0], datanew.shape[1], datanew.shape[2]))

    lib.warning("  *** patch_projection")
    slider(np.log(np.abs(fdatanew.swapaxes(0,1))), axis=0)
    a = np.real(np.fft.ifft(fdatanew,axis=2))
    b = np.imag(np.fft.ifft(fdatanew,axis=2))
    # print(a.shape)
    slider(a.swapaxes(0,1), axis=0)
    slider(b.swapaxes(0,1), axis=0)
    return np.real(np.fft.ifft(fdatanew,axis=2))


def guess_center(first_projection, last_projection):
    """
    Compute the tomographic rotation center based on cross-correlation technique.
    *first_projection* is the projection at 0 deg, *last_projection* is the
    projection at 180 deg.
    """
    from scipy.signal import fftconvolve
    width = first_projection.shape[1]
    first_projection = first_projection - first_projection.mean()
    last_projection = last_projection - last_projection.mean()

    # The rotation by 180 deg flips the image horizontally, in order
    # to do cross-correlation by convolution we must also flip it
    # vertically, so the image is transposed and we can apply convolution
    # which will act as cross-correlation
    convolved = fftconvolve(first_projection, last_projection[::-1, :], mode='same')
    center = np.unravel_index(convolved.argmax(), convolved.shape)[1]

    return (width / 2.0 + center) / 2


class CenterCalibration(object):

    def __init__(self, first, last):
        self.center = guess_center(first, last)
        self.height, self.width = first.shape

    @property
    def position(self):
        return self.width / 2.0 + self.width - self.center * 2.0

    @position.setter
    def position(self, p):
        self.center = (self.width / 2.0 + self.width - p) / 2


