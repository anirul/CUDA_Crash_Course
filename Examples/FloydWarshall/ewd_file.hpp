#ifndef EWD_FILE_HEADER_DEFINED
#define EWD_FILE_HEADER_DEFINED

#include <iostream>
#include <fstream>
#include <map>
#include <string>

const float huge_float = 1e30f;

class ewd_file {
protected :
	size_t nb_vector_;
	size_t nb_edges_;
	std::map<std::pair<unsigned int, unsigned int>, float> edges_;
public :
	ewd_file();
	virtual ~ewd_file();
	void import_file(const std::string& name);
	void export_file(const std::string& name);
	void import_matrix(float* p, size_t size);
	void export_matrix(float* p, size_t size);
	size_t size() const;
	float dist(unsigned int v1, unsigned int v2) const;
   void print_matrix(std::ostream& os);
	std::ostream& operator<<(std::ostream& os);
};

#endif

