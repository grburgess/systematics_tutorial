import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import gamma

class Fit(object):

    def __init__(self, data, samples):

        self._data = data
        self._samples = samples

    def plot(self,data_color='b',latent_color='g', fit_color='y'):

        fig, ax = plt.subplots()

        # first plot the data and latent

        fig = self._data.plot(ax=ax, with_latent=True, data_color=data_color, latent_color=latent_color)

        # now plot the fit
        x_values = self._data.x_latent

        # the first parameter is always the slope parameter
        for m in self._samples[::100, 0]:

            fit_line = self._data.line(x_values, m, self._data.b_latent)

            ax.plot(x_values, fit_line, color=fit_color, alpha=0.05, zorder=-100)

        return fig

    def compute_m_quantiles(self, cr=90):

        lower = 50 - float(cr) / 2.
        upper = 50 + float(cr) / 2.

        lower_cr, upper_cr = np.percentile(self._samples[:, 0], q=[lower, upper])

        return (lower_cr, upper_cr)

    def check_success(self, cr=90):

        l, u = self.compute_m_quantiles(cr)

        if (self._data.m_latent>= l) & (self._data.m_latent<=u):

            return 1.

        else:

            return 0.

    @property
    def samples(self):
        return self._samples
    
    def density_plot(self, cr=None, ax=None, with_latent=True, color='b', cr_color='r', cr_alpha=0.1, latent_color='w', line_alpha=1., height=0):

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        density = gaussian_kde(self._samples[:, 0])

        xplot = np.linspace(self._samples[:, 0].min(), self._samples[:, 0].max(), 200)

        ax.plot(xplot, density(xplot)  + height, color=color, alpha=line_alpha)

        if cr is not None:
            l, u = self.compute_m_quantiles(cr)
            xplot = np.linspace(l, u, 200)

            ax.fill_between(xplot, height, density(xplot) + height , color=cr_color, alpha=cr_alpha)

        if with_latent:

            ax.axvline(self._data.m_latent, ls='--', color=latent_color)

            
        ax.set_xlabel('m')
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        return fig


    

class Fitter(object):

    def __init__(self, data_generator, latent_params, N=100, ndim=1):

        self._data_generator = data_generator
        self._ndim = ndim

        # it is assumed the first two latent params are m_latent
        # and b_latent.

        # the third pararmeter is the sigma.

        self._latent_params = latent_params

        self._m_latent = latent_params[0]
        self._b_latent = latent_params[1]
        self._sigma = latent_params[2]

        self._N = N

        self._nwalkers = 100

        self._fits = []

    def perform_fit(self):

        # generate some data

        data = self._data_generator(*self._latent_params)

        # define the log probability
        def lnprob(theta, x, y, yerr):
            lp = self._lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + self._lnlike(theta, x, y, yerr)

        # get some initial positions

        pos = [np.array([2] * self._ndim) + 1e-4 * np.random.randn(self._ndim) for i in range(self._nwalkers)]

        # create the sampler
        sampler = emcee.EnsembleSampler(
            self._nwalkers, self._ndim, lnprob, args=(data.x_latent, data.y_observed, data.sigma))
        # run the sampler
        sampler.run_mcmc(pos, 500)

        # extract the sampler
        samples = sampler.chain[:, 50:, :].reshape((-1, self._ndim))

        self._fits.append(Fit(data, samples))

    def check_confidence_regions(self, N=100,cr=90):

        # clear out the past fits
        self.reset()

        # perform N fits

        for i in range(int(N)):

            self.perform_fit()

        successes = sum([x.check_success(cr) for x in self._fits])


        fractional_success = float(successes) / float(N)

        print("%d of %d fits (%f) were successfully inside the %f %% confidence region" %(successes,
                                                                                           N,
                                                                                           fractional_success,
                                                                                           cr))
        
        if cr/100. <= fractional_success:

           print('There are no systematics') 

        else:

           
            print('There are systematics!')
            

            

            
        
    @property
    def fits(self):
        return self._fits

    def reset(self):
        """
        Dump all of the past fits
        """

        self._fits = []


class NormalFit(Fitter):

    def __init__(self, data_generator, latent_params, N=100):

        super(NormalFit, self).__init__(data_generator, latent_params, N, 1)

        # define a prior for the slope only

        def lnprior(theta):
            m = theta

            if (-5.0 < m < 5.0):
                return 0.0
            return -np.inf

            
            return (-1.5) * np.log(1+m**2)

        # define a gaussian likelihood for the slope only
        def lnlike(theta, x, y, yerr):
            m = theta
            model = m * x + self._b_latent
            inv_sigma2 = 1.0 / (yerr**2)
            return -0.5 * (np.sum((y - model)**2 * inv_sigma2))

        self._lnprior = lnprior
        self._lnlike = lnlike


class NormalSysFit(Fitter):

    def __init__(self, data_generator, latent_params, N=100, systematic=.5):

        super(NormalSysFit, self).__init__(data_generator, latent_params, N, 1)

        # define a prior for the slope only
        self._systematic = systematic
        
        def lnprior(theta):
            m = theta

            if (-5.0 < m < 5.0):
                return 0.0
            return -np.inf

            
            return (-1.5) * np.log(1+m**2)

        # define a gaussian likelihood for the slope only
        def lnlike(theta, x, y, yerr):
            m = theta

            # add on the systematic
            
            
            model = m * x + self._b_latent
            inv_sigma2 = 1.0 / np.sqrt(yerr**2 + self._systematic**2)
            return -0.5 * (np.sum((y - model)**2 * inv_sigma2))

        self._lnprior = lnprior
        self._lnlike = lnlike

        

class NormalSigmaFit(Fitter):

    def __init__(self, data_generator, latent_params, N=100):

        super(NormalSigmaFit, self).__init__(data_generator, latent_params, N, 2)

        # define a prior for the slope AND the sigma of the data

        def lnprior(theta):
            m, sigma = theta
            
            if sigma < 0:
                return -np.inf  # log(0)
            else:
                return -1.5 * np.log(1 + m ** 2) - np.log(sigma)
        
        # define a gaussian likelihood for the slope and the
        # the sigma. Note that yerr is ignored
        def lnlike(theta, x, y, yerr):
            m, sigma = theta

            model = m * x + self._b_latent

            inv_sigma2 = 1.0 / (sigma**2)

            return -0.5 * (np.sum((y - model)**2 * inv_sigma2))

        self._lnprior = lnprior
        self._lnlike = lnlike


class TFit(Fitter):

    def __init__(self, data_generator, latent_params, N=100):

        super(TFit, self).__init__(data_generator, latent_params, N, 2)

        # define a prior for the slope AND the sigma of the data

        def lnprior(theta):
            m, nu = theta
            if (-5.0 < m < 5.0) and (2 <= nu < 10.):
                return 0.0
            return -np.inf

        # define a gaussian likelihood for the slope and the
        # the sigma. Note that yerr is ignored
        def lnlike(theta, x, y, yerr):
            m, nu = theta
            model = m * x + self._b_latent
            inv_sigma2 = 1.0 / (yerr**2 *(nu-2))
            ss = (np.sum(1 + (y - model)**2 * inv_sigma2))

            factor = len(x) * np.log(gamma((nu+1.)/2.)/( np.sqrt(np.pi *(nu-2)) * gamma(0.5 * nu))  )
            return  factor + -0.5*(nu+1) * ss

        self._lnprior = lnprior
        self._lnlike = lnlike

class CauchyFit(Fitter):

    def __init__(self, data_generator, latent_params, N=100):

        super(CauchyFit, self).__init__(data_generator, latent_params, N, 1)

        # define a prior for the slope only

        def lnprior(theta):
            m = theta

            if (-5.0 < m < 5.0):
                return 0.0
            return -np.inf

            

        # define a gaussian likelihood for the slope only
        def lnlike(theta, x, y, yerr):
            m = theta
            model = m * x + self._b_latent

            gamma2 = yerr**2

            return -np.sum(np.log( gamma2 +  (y - model)**2 ))

        self._lnprior = lnprior
        self._lnlike = lnlike
