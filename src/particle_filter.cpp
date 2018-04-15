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

static default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	normal_distribution<double> noise_x(0.0, std[0]);
	normal_distribution<double> noise_y(0.0, std[1]);
	normal_distribution<double> noise_theta(0.0, std[2]);

	for (unsigned int i=0; i<num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1.0;

		// Add noise
		p.x += noise_x(gen);
		p.y += noise_y(gen);
		p.theta += noise_theta(gen);

		particles.push_back(p);
	}

	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	for (unsigned int i=0; i<num_particles; i++) {
		// Predict the new state
		if (0.001 < fabs(yaw_rate)) {
			particles[i].x += velocity / yaw_rate * ( sin(particles[i].theta+yaw_rate*delta_t) - sin(particles[i].theta) );
			particles[i].y += velocity / yaw_rate * ( cos(particles[i].theta) - cos(particles[i].theta+yaw_rate*delta_t) );
			particles[i].theta += yaw_rate * delta_t;
		} else {
			particles[i].x += velocity * cos(particles[i].theta) * delta_t;
			particles[i].y += velocity * sin(particles[i].theta) * delta_t;
		}

		// Add noise
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);
	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// Nearest-neighbors
	for (unsigned int i=0; i<observations.size(); i++) {
		double min_dist = numeric_limits<double>::max();
		int map_id = -1;

		for (unsigned int j=0; j<predicted.size(); j++) {
			double current_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (current_dist < min_dist) {
				min_dist = current_dist;
				map_id = predicted[j].id;
			}
		}

		observations[i].id = map_id;
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

	for (unsigned int i=0; i<num_particles; i++) {

		// Predicted landmarks are only within the range of sensor
		std::vector<LandmarkObs> predicted;
		for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++) {
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;

			if (dist(particles[i].x, particles[i].y, landmark_x, landmark_y) < sensor_range) {
			  predicted.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
			}
		}

		// Transform from vehicle coordinates to map coordinates
		std::vector<LandmarkObs> transformed_obs;
		for (unsigned int j=0; j<observations.size(); j++) {
			double transformed_x = observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta) + particles[i].x;
			double transformed_y = observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta) + particles[i].y;

			transformed_obs.push_back(LandmarkObs{observations[j].id, transformed_x, transformed_y});
		}

		//Nearest-neighbors
		dataAssociation(predicted, transformed_obs);

		// Reinit the weight
		particles[i].weight = 1.0;
		for (unsigned int j=0; j<transformed_obs.size(); j++) {
			// Find the predicted landmark whose id is same as the observed landmark
			unsigned int idx = 0;
			while (predicted[idx].id != transformed_obs[j].id) {
				idx++;
			}

			particles[i].weight *= 1.0 / (2.0*M_PI*std_landmark[0]*std_landmark[1]) * exp(
														   -(pow(transformed_obs[j].x-predicted[idx].x,2)/(2.0*pow(std_landmark[0],2))
															   +pow(transformed_obs[j].y-predicted[idx].y,2)/(2.0*pow(std_landmark[1],2))));
		}
	}
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Init discrete distribution with particles' weights
	std::vector<double> weights;
	for (unsigned int i=0; i<num_particles; i++) {
		weights.push_back(particles[i].weight);
	}
	std::discrete_distribution<int> dist{weights.begin(), weights.end()};

	// Resampling
	std::vector<Particle> tmp_particles;
	for (unsigned int i=0; i<num_particles; i++) {
		tmp_particles.push_back(particles[dist(gen)]);
	}

	particles = tmp_particles;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
