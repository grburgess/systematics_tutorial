data {
  int<lower=0> N; // number of measurments
  vector[N] y; // repeated observations
}
parameters {
  real mu; // mean
  real<lower = 0> sigma; //width
}

model {
  // measurement model
  // uninformative priors on mu and sigma
  y ~ normal(mu, sigma);  
  
}
