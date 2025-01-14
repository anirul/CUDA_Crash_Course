#include <limits>
#include <stdexcept>
#include <sstream>
#include <utility>
#include <cmath>
#include "ewd_file.hpp"

ewd_file::ewd_file() : nb_vector_(0), nb_edges_(0) {
   edges_.clear();
}

ewd_file::~ewd_file() {
   nb_vector_ = 0;
   nb_edges_ = 0;
   edges_.clear();
}

void ewd_file::import_file(const std::string& name)
{
   std::ifstream ifs;
   ifs.open(name.c_str());
   if (!ifs.is_open()) 
      throw std::runtime_error("Could not open file : " + name);
   ifs >> nb_vector_;
   if (nb_vector_ == 0)
      throw std::runtime_error("Parse error invalid vectors size");
   ifs >> nb_edges_;
   if (nb_edges_ == 0)
      throw std::runtime_error("Parse error invalid edges size");
   std::cout << "Graph size      : " << nb_vector_ << std::endl;
   for (int i = 0; i < nb_edges_; ++i) {
      std::cout << "Graph edge [" << (i + 1) << "/" << nb_edges_ << "]\r";
      std::cout.flush();
      std::stringstream ss("");
      ss << "Parse error at edge : " << i;
      unsigned int v1;
      unsigned int v2;
      float d;
      ifs >> v1;
      ifs >> v2;
      ifs >> d;
      if (v1 >= nb_vector_) {
         ss << " invalid first vector : " << v1;
         throw std::runtime_error(ss.str());
      }
      if (v2 >= nb_vector_) {
         ss << " invalid second vector : " << v2; 
         throw std::runtime_error(ss.str());
      }
      if (d <= 0.0f) {
         ss << " invalid distance : " << d;
         throw std::runtime_error(ss.str());
      }
      edges_.insert(std::make_pair(std::make_pair(v1, v2), d));
   }
   std::cout << std::endl;
   ifs.close();
}

void ewd_file::export_file(const std::string& name)
{
   std::ofstream ofs;
   ofs.open(name.c_str());
   if (!ofs.is_open())
      throw std::runtime_error("Could not open file : " + name);
   this->operator<<(ofs);
   ofs.close();
}

float ewd_file::dist(unsigned int v1, unsigned int v2) const {
   if (v1 == v2) 
      return 0.0f;
   std::map<std::pair<unsigned int, unsigned int>, float>::const_iterator ite;
   ite = edges_.find(std::make_pair(v1, v2));
   if (ite == edges_.end()) 
      return huge_float;
   return ite->second;
}

size_t ewd_file::size() const {
   return nb_vector_;
}

void ewd_file::import_matrix(float* p, size_t size)
{
   nb_vector_ = static_cast<size_t>(std::sqrt(size));
   for (int x = 0; x < nb_vector_; ++x) {
      for (int y = 0; y < nb_vector_; ++y) {
         float distance = p[x + (y * nb_vector_)];
         if (distance < huge_float) {
            edges_.insert(std::make_pair(std::make_pair(x, y), distance));
         }
      }
   }
}

void ewd_file::export_matrix(float* p, size_t size)
{
   if ((nb_vector_ * nb_vector_) != size)
      throw std::runtime_error("Unmatched size!");
   for (int x = 0; x < nb_vector_; ++x) {
      std::cout 
         << "Export matrix line [" << x + 1 << "/" << nb_vector_ << "]\r";
      std::cout.flush();
      for (int y = 0; y < nb_vector_; ++y) {
         p[x + (y * nb_vector_)] = dist(x, y);
      }
   }
   std::cout << std::endl;
}

void ewd_file::print_matrix(std::ostream& os) {
   for (int x = 0; x < nb_vector_; ++x) {
      for (int y = 0; y < nb_vector_; ++y) {
         std::stringstream ss("");
         ss << dist(x, y);
         int line_left = 8 - static_cast<int>(ss.str().size());
         os << ss.str();
         for (int i = 0; i < line_left; ++i)
            os << " ";
      }
      std::cout << std::endl;
   }
}

std::ostream& ewd_file::operator<<(std::ostream& os) {
   os << nb_vector_ << std::endl;
   os << nb_edges_ << std::endl;
   std::map<std::pair<unsigned int, unsigned int>, float>::const_iterator ite;
   for (ite = edges_.begin(); ite != edges_.end(); ++ite) {
      os 
         << ite->first.first << "\t" 
         << ite->first.second << "\t"
         << ite->second << std::endl;
   }
   return os;
}

