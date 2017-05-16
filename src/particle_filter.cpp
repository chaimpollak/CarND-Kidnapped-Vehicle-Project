/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */


#include <random>
#include <algorithm>
#include <iostream>

#include "map.h"
#include "particle_filter.h"

using namespace std;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // I started off using 1000 particles, but narrowed it down to 100 based on the runtime
  // I was able to narrow it down to 50 and still get good results but I felt that it is too risky to
  // use only 50 particles
  num_particles = 100;

  default_random_engine gen;
  normal_distribution<double>  dist_x(x, std[0]);// std[0] => standard deviation of x [m]
  normal_distribution<double>  dist_y(y, std[1]);// std[1] => standard deviation of y [m]
  normal_distribution<double>  dist_theta(theta, std[2]);// std[2] => standard deviation of yaw [rad]


  for (int i = 0; i < num_particles; i++)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;

    particles.push_back(particle);
    weights.push_back(1);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {

    double predicted_x,predicted_y,predicted_theta;

    // get prediction
    if (yaw_rate == 0) {
      predicted_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
      predicted_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
      predicted_theta = particles[i].theta;
    } else {
      predicted_x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
      predicted_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
      predicted_theta = particles[i].theta + yaw_rate*delta_t;
    }

    // add noise
    normal_distribution<double> dist_x(predicted_x, std_pos[0]);// std_pos[0] => standard deviation of x [m]
    normal_distribution<double> dist_y(predicted_y, std_pos[1]);// std_pos[1] => standard deviation of y [m]
    normal_distribution<double> dist_theta(predicted_theta, std_pos[2]);// std_pos[2] => standard deviation of yaw [rad]

    // set particle measurement
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);

  }


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> &observations, std::vector<Map::single_landmark_s>  landmarks) {

  // for every observation find the nearest landmark
  for (int i = 0; i < observations.size(); i++){
    double min_distance = 99999;
    int nearest_index = -1;
    for(int j = 0;j< landmarks.size();j++){
      double distance = sqrt(pow(observations[i].x - landmarks[j].x_f,2) + pow(observations[i].y- landmarks[j].y_f,2));
      if(distance < min_distance){
        min_distance = distance;
        nearest_index = j;
      }
    }
    //set association for observation[i] to landmark[predicted_index]
    observations[i].id = nearest_index;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  for (int i = 0; i < num_particles; i++){

    // reset the particle weight to 1
    particles[i].weight = 1.0;

    // transform the observations
    vector<LandmarkObs> transformed_landmarks =  transform_landmarks(particles[i], observations);


    // set associations between the map landmarks and the observations
    dataAssociation(transformed_landmarks, map_landmarks.landmark_list);

    // update particle weight
    for(int j = 0;j < transformed_landmarks.size();j++) {
      Map::single_landmark_s nearest_landmark = map_landmarks.landmark_list[transformed_landmarks[j].id];
      double observation_x = transformed_landmarks[j].x;
      double observation_y = transformed_landmarks[j].y;
      double landmark_x = nearest_landmark.x_f;
      double landmark_y =  nearest_landmark.y_f;

      double x_diff = pow(observation_x - landmark_x, 2.0);
      double y_diff =  pow(observation_y - landmark_y, 2.0);
      double std_range = 2 * pow(std_landmark[0], 2) ;
      double std_bearing = 2 * pow(std_landmark[1], 2);

      double norm_weight = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]) * exp( -( x_diff /std_range + y_diff /std_bearing ));

      if(norm_weight > 0) {
        particles[i].weight *= norm_weight; // should
      }

    }
    // update the weight
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {

  default_random_engine gen;

  discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> p2;

  for (int i = 0; i < num_particles; i++) {
    p2.push_back(particles[distribution(gen)]);
  }

  particles = p2;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

vector<LandmarkObs> ParticleFilter::transform_landmarks(Particle particle, std::vector<LandmarkObs> landmark_observations){
  vector<LandmarkObs> transformed_landmarks;

  for (int i = 0; i < landmark_observations.size(); i++) {
    float transformed_x = landmark_observations[i].x * cos(particle.theta) - landmark_observations[i].y * sin(particle.theta);
    float transformed_y = landmark_observations[i].x * sin(particle.theta) + landmark_observations[i].y * cos(particle.theta);

    LandmarkObs transformed_landmark;
    transformed_landmark.x = particle.x + transformed_x;
    transformed_landmark.y = particle.y + transformed_y;
    transformed_landmarks.push_back(transformed_landmark);
  }
  return transformed_landmarks;
}

