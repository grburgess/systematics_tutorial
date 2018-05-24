functions {

  real lightcurve(real time, real idx1, real peak_flux, real peak_time, real idx2) {
    
    if(time <= peak_time) {
      
      return peak_flux * pow(time/peak_time, idx1);
    }
    
    else {
      
      return peak_flux * pow(time/peak_time, idx2);
    }
    
  }

}



data {

  int N ;
  vector[N] flux_obs;
  real flux_sigma;
  vector[N] time;
  real calibration_mu;
  real calibration_sigma;
  int N_model;
  vector[N_model] time_model;


}

parameters {

  real<lower=0> peak_flux;
  real<lower=0> peak_time;

  real idx1;
  real idx2;

  real calibration;


}

transformed parameters {

  vector[N] true_intensity;
  vector[N] flux_latent;



  for (n in 1:N) {

    true_intensity[n] = lightcurve(time[n], idx1, peak_flux, peak_time, idx2);
  }

  flux_latent = calibration + true_intensity; 
  

}

model {
  
  // parameters
  idx1 ~ normal(0, 2);
  idx2 ~ normal(-2, 2);

  peak_flux ~ lognormal(log(100), 2);
  peak_time ~ lognormal(log(100), 2);

  calibration ~ normal(calibration_mu, calibration_sigma);


  flux_obs ~ normal(flux_latent, flux_sigma);
  

}

generated quantities {

  vector[N_model] lc_fit;


  for (n in 1:N_model) {

    lc_fit[n] = lightcurve(time_model[n], idx1, peak_flux, peak_time, idx2);

  }


}
