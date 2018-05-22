data {
  int<lower=0> N; // number of measurements
  vector[N] y; // repeated measurements
 }
parameters {
  real mu; //mean
  real<lower=0> sigma_sq; // width
  vector<lower=-0.5, upper=0.5>[N] y_err; // measurement error from rounding
}
transformed parameters {
  real<lower=0> sigma;
  vector[N] z;
  
  sigma = sqrt(sigma_sq);
  // the latent value of the measurement
  z = y + y_err;
}
model {

  // measurement model
  target += -2 * log(sigma);
  z ~ normal(mu, sigma);
}
