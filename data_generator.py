import numpy as np
import scipy.stats as stats
import scipy.special as sf
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as op
import matplotlib.pyplot as plt


class DataBuilder(object):
    SIZE = 20
    x_values = np.random.uniform(-5,5, size=SIZE)
    
    def __init__(self, m_latent, b_latent):


        self._m_latent = m_latent
        self._b_latent = b_latent
        
        self._x_values = DataBuilder.x_values
        self._y_latent = self._gen_y_latent()
        self._y_observed = self._generate_y_observed()

    @property
    def y_observed(self):
        return self._y_observed
    
    @property
    def y_latent(self):
        return self._y_latent
    
    @property
    def x_latent(self):
        return self._x_values

    @property
    def sigma(self):
        return self._sigma

    @property
    def m_latent(self):
        return self._m_latent

    
    @property
    def b_latent(self):
        return self._b_latent

    
    def line(self, x, m, b):
        '''
        The line function
        '''
    
        return m*x + b

    def _gen_y_latent(self):

        y_latent = self.line(self._x_values, self._m_latent, self._b_latent) 

        return y_latent

    def _generate_y_observed(self):


        y_observed = self._y_latent + self._dist.rvs(size=DataBuilder.SIZE)

        return y_observed

    def plot(self, ax = None, with_latent=False, latent_color='r', data_color='b'):

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()
            
        ax.errorbar(self._x_values, self._y_observed, yerr=self._sigma, fmt='.',color=data_color)

        if with_latent:

            ax.plot(self._x_values, self.line(self._x_values,self._m_latent, self._b_latent),'-', color=latent_color)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        return fig



class NormalData(DataBuilder):

    def __init__(self, m_latent, b_latent, sigma):

        self._sigma = sigma
        self._dist = stats.norm(scale=sigma, loc=0)

        super(NormalData, self).__init__(m_latent, b_latent)

class TData(DataBuilder):
    
    def __init__(self, m_latent, b_latent, sigma, nu):

        self._sigma = sigma
        self._nu = nu
        self._dist = stats.t(df=nu, scale=sigma, loc=0)

        super(TData, self).__init__(m_latent, b_latent)



class CauchyData(DataBuilder):
    
    def __init__(self, m_latent, b_latent, sigma):

        self._sigma = sigma

        self._dist = stats.cauchy(scale=sigma, loc=0)

        super(CauchyData, self).__init__(m_latent, b_latent)






