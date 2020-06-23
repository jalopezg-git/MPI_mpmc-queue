# MPI MPMC bounded queue implementation
This repository contains the Multiple-Producer-Multiple-Consumer distributed queue used in the GrPPI MPI back-end. It is internally based on the `MPI_Fetch_and_op()`, `MPI_Compare_and_swap()`, `MPI_Get()` and `MPI_Put()` one-sided (RMA) calls.

This queue supports pushing elements of any type. The only requirement is that serialization/deserialization is supported through boost::archive.

An example of its use is included below:
```
#include <iostream>

#include "std_optional_serialize.h"
#include "mpi_mpmc_queue.h"

#define CAPACITY 333
#define DATA_AREA_SZ 4096000

int main(int argc, char *argv[]) {
	int rank, prov;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &prov);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// In this example, roles are assigned depending on the value of `rank`.
	//
	// If the rank of a process is an even number, it will be a producer;
	// otherwise, it will be a consumer.
	grppi::mpi_mpmc_queue<std::pair<std::experimental::optional<std::string>,long>> q{CAPACITY, DATA_AREA_SZ, 0,
		(rank == 0) ? grppi::mpi_mpmc_role::CONTROLLER 
			: static_cast<grppi::mpi_mpmc_role>((rank & 0x01) + 1)};

	char vec[4] = { 0x01, 0x01, 0x01, 0x01 };
	char (*tab)[4] = new char[4][4];
	grppi::internal::make_adj_matrix(vec, tab);
	q.enable(tab, 0);

	if (rank & 0x01) {
		while (true) {
			auto e = q.pop();
			if (!e.first) break;
			std::cout << "pop() at " << rank << ": " << e.first.value() << std::endl;
		}
	} else {
		std::string s = "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Ut purus elit, vestibulum ut, placerat ac, adipiscing vitae, felis.";

		q.push(std::make_pair(std::experimental::optional<std::string>{s}, 0));
		std::cout << "push() at " << rank << ": " << s << std::endl;

		// send end-of-stream (EOS) element
		q.push(std::make_pair(std::experimental::optional<std::string>{}, 0));
	}

	return 0;
}
```
