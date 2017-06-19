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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
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

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
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

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
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

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

void ParticleFilter::resample() {
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
