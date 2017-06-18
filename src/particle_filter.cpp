/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

/**
 * init Initializes particle filter by initializing particles to Gaussian
 *   distribution around first position and all the weights to 1.
 * @param x Initial x position [m] (simulated estimate from GPS)
 * @param y Initial y position [m]
 * @param theta Initial orientation [rad]
 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
 *   standard deviation of yaw [rad]]
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 100;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; ++i) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;
		// Add Particle and weights
		particles.push_back(particle);
		weights.push_back(particle.weight);
	}
	// Done Setting up
	is_initialized = true;
}

/**
 * prediction Predicts the state for the next time step
 *   using the process model.
 * @param delta_t Time between time step t and t+1 in measurements [s]
 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
 *   standard deviation of yaw [rad]]
 * @param velocity Velocity of car from t to t+1 [m/s]
 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	default_random_engine gen;

	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) > 0.001) {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		} else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
			// No change in theta
		}
		// This line creates a normal (Gaussian) distribution for x
		normal_distribution<double> dist_x(particles[i].x, std_x);
		normal_distribution<double> dist_y(particles[i].y, std_y);
		normal_distribution<double> dist_theta(particles[i].theta, std_theta);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

/**
 * dataAssociation Finds which observations correspond to which landmarks (likely by using
 *   a nearest-neighbors data association).
 * @param predicted Vector of predicted landmark observations
 * @param observations Vector of landmark observations
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
		LandmarkObs observation = observations[i];
		double min_dist = std::numeric_limits<double>::max();

		for (int j = 0; j < predicted.size(); j++) {
			// From Helper Functions
			double distance = dist(observation.x,
															observation.y,
															predicted[j].x,
															predicted[j].y);
			if (distance < min_dist) {
				min_dist = distance;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

/**
 * updateWeights Updates the weights for each particle based on the likelihood of the
 *   observed measurements.
 * @param sensor_range Range [m] of sensor
 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
 *   standard deviation of bearing [rad]]
 * @param observations Vector of landmark observations
 * @param map Map class containing map landmarks
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double coeff = 1 / (2 * M_PI * std_x * std_y);
	double x_denom = 2 * std_x * std_x;
	double y_denom = 2 * std_y * std_y;
	double x_numer;
	double y_numer;

	for (int i = 0; i < num_particles; i++) {
		vector<LandmarkObs> translated_landmarks;
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs translated_landmark;
			translated_landmark.x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
			translated_landmark.y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
			translated_landmarks.push_back(translated_landmark);
		}

		// Create list of predicted landmarks
		vector<LandmarkObs> predicted_landmarks;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

			if (distance <= sensor_range) {
				LandmarkObs predicted_landmark;
				predicted_landmark.x = map_landmarks.landmark_list[j].x_f;
				predicted_landmark.y = map_landmarks.landmark_list[j].y_f;
				predicted_landmark.id = map_landmarks.landmark_list[j].id_i;
				predicted_landmarks.push_back(predicted_landmark);
			}
		}

		// User helper function
		dataAssociation(predicted_landmarks, translated_landmarks);

		double new_weight = 1.0;

		for (int j = 0; j < translated_landmarks.size(); j++) {
			//The mean of the Multivariate-Gaussian is the measurement's associated landmark position
			double u_x;
			double u_y;
			for (int k = 0; k < predicted_landmarks.size(); k++) {
				if (translated_landmarks[j].id == predicted_landmarks[k].id) {
					u_x = predicted_landmarks[k].x;
					u_y = predicted_landmarks[k].y;
				}
			}
			x_numer = (translated_landmarks[j].x - u_x) * (translated_landmarks[j].x - u_x);
			y_numer = (translated_landmarks[j].y - u_y) * (translated_landmarks[j].y - u_y);
			new_weight *= coeff * exp(-(x_numer/x_denom) - (y_numer/y_denom));
			particles[i].weight = new_weight;
		}
		weights[i] = particles[i].weight;
	}
}

/**
 * resample Resamples from the updated set of particles to form
 *   the new set of particles.
 */
void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
		std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> weights_rand(0,num_particles - 1);
		int random_weight_index = weights_rand(gen);

		double max_weight = *max_element(weights.begin(), weights.end());
		double beta = 0.0;

		default_random_engine beta_gen;
		uniform_real_distribution<double> beta_rand(0.0, (2 * max_weight));

		std::vector<Particle> resampled_particles;

		for(int i = 0; i < num_particles; i++) {
			beta += beta_rand(beta_gen);
			while(weights[random_weight_index] < beta) {
				beta -= weights[random_weight_index];
				random_weight_index = (random_weight_index + 1) % num_particles;
			}
			resampled_particles.push_back(particles[random_weight_index]);
		}
		particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
