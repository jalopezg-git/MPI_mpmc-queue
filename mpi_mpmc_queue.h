/**
* @version		GrPPI v0.2
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef GRPPI_COMMON_MPI_MPMC_QUEUE_GP_H
#define GRPPI_COMMON_MPI_MPMC_QUEUE_GP_H

#include <mpi.h>
#include <boost/serialization/utility.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <utility>
#include <vector>
#include <type_traits>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include <sstream>
#include <mutex>
#include <experimental/optional>

//#include "queue_traits.h"
#include "std_optional_serialize.h"

namespace grppi{
namespace internal {
	MPI_Group  MPI_GROUP_WORLD = MPI_GROUP_NULL;
	MPI_Datatype  MPI_3INT = MPI_DATATYPE_NULL;
	std::mutex mutex_mpi; // OpenMPI does not support MPI_THREAD_MULTIPLE + RMA over pt2pt
	                      // MPI_THREAD_SERIALIZED + mutex is mandatory
	// ELF ctor/dtor
	void __attribute__((destructor(199)))
	__ELF_dtor__() { if (MPI_GROUP_WORLD != MPI_GROUP_NULL)
				MPI_Group_free(&MPI_GROUP_WORLD);
			 if (MPI_3INT != MPI_DATATYPE_NULL)
				MPI_Type_free(&MPI_3INT);
			 MPI_Finalize(); }

	template <typename U, typename V,
		 typename = std::enable_if_t<std::is_array<U>::value>,
		 typename = std::enable_if_t<std::is_pointer<V>::value>>
	void make_adj_matrix(const U& my_bitmap, V& adj) {
	        // Build a rank-queue adjacency matrix, e.g.
	        //  rank \ queue
	        //        \---------------------------
	        //      0 |  0  1  1  1  0  1  1  0  |
	        //      1 |  1  1  1  0  1  0  1  1  |
	        //      . |  .  .  .  .  .  .  .  .  |
	        //      2 |  1  1  0  1  1  1  0  1  |
	        //        ----------------------------
	        MPI_Allgather(&my_bitmap, sizeof(my_bitmap), MPI_BYTE, adj,
	       		 sizeof(my_bitmap), MPI_BYTE, MPI_COMM_WORLD);
	}
}
enum class mpi_mpmc_role { CONTROLLER = 0, PRODUCER, CONSUMER };


template <typename T>
class mpi_mpmc_queue{
   public:
      using value_type = T;
      using iarchive_type = boost::archive::binary_iarchive;
      using oarchive_type = boost::archive::binary_oarchive;
	
      mpi_mpmc_queue<T>(int _capacity, int _data_size, int _rank_controller,
		      mpi_mpmc_role _role = mpi_mpmc_role::CONSUMER) :
	      capacity{_capacity}, data_size{_data_size}, role{_role}, data{NULL}, ctl{NULL},
	      _offset_bitmap{9 + (_capacity * 3)}, w_data{MPI_WIN_NULL}, w_ctl{MPI_WIN_NULL},
	      rank_controller{_rank_controller}
      {}
      ~mpi_mpmc_queue() {
	      // MPI_Barrier(comm) should not be required, as MPI_Win_free() is a collective call
	      MPI_Win_free(&w_data);
	      MPI_Win_free(&w_ctl);
	      MPI_Type_free(&MPI_T_BITMAP);
	      MPI_Comm_free(&comm);
	      delete [] data;
	      delete [] ctl;
      }

      mpi_mpmc_queue(mpi_mpmc_queue && q) :
	      capacity{q.capacity}, data_size{q.data_size}, n_members{q.n_members}, role{q.role},
	      m{}, comm{q.comm}, w_data{q.w_data}, w_ctl{q.w_ctl}, MPI_T_BITMAP{q.MPI_T_BITMAP},
	      my_rank{q.my_rank}, enabled{q.enabled.load()} 
      {
	      std::swap(data, q.data);
	      std::swap(ctl, q.ctl);
	      _offset_bitmap = q._offset_bitmap;
	      rank_controller = q.rank_controller;
      }

      mpi_mpmc_queue(const mpi_mpmc_queue &) = delete; 
      mpi_mpmc_queue & operator=(const mpi_mpmc_queue &) = delete;


      template <typename U,
                typename = std::enable_if_t<std::is_pointer<U>::value>>
      void enable(U& adj, int q) {
	      std::vector<int> ranks;
	      MPI_Group grp_this;
	      int procs, tmp[1];

	      MPI_Comm_size(MPI_COMM_WORLD, &procs);
	      if (internal::MPI_GROUP_WORLD == MPI_GROUP_NULL)
		      MPI_Comm_group(MPI_COMM_WORLD, &internal::MPI_GROUP_WORLD);

	      // Uses MPI_Allgather(), instead of MPI_Comm_split() to make MPI -see make_adj_matrix()-
	      // communicator of processes that reference this queue
	      for (int i = 0; i < procs; ++i)
		      if (adj[i][q >> 3] & (1 << (q & 0x07)))
			      ranks.push_back(i);
	      MPI_Group_incl(internal::MPI_GROUP_WORLD, ranks.size(), ranks.data(), &grp_this);
	      MPI_Comm_create_group(MPI_COMM_WORLD, grp_this, q, &comm);

	      MPI_Comm_rank(comm, &my_rank);
	      MPI_Group_translate_ranks(internal::MPI_GROUP_WORLD, 1, rank_controller, grp_this, tmp);
	      std::swap(tmp[0], rank_controller[0]);

	      MPI_Group_size(grp_this, &n_members);
	      MPI_Group_free(&grp_this);

	      // RMA: register memory regions
	      auto ctl_size = get_ctl_region_size();
	      if (role != mpi_mpmc_role::CONSUMER) {
		      data = new char[data_size], ctl = new char[ctl_size];
		      memset(ctl, 0, ctl_size);
	      }
	      _offset_bitmap[1] = _offset_bitmap[0] + bitmap_size();
	      _offset_bitmap[2] = _offset_bitmap[1] + bitmap_size();

	      MPI_Win_create(data, (role != mpi_mpmc_role::CONSUMER) ? data_size : 0, 1,
			      MPI_INFO_NULL, comm, &w_data);
	      MPI_Win_create(ctl, ctl_size, sizeof(int), MPI_INFO_NULL, comm, &w_ctl);

	      // Required datatypes
	      MPI_Type_contiguous(bitmap_size(), MPI_INT, &MPI_T_BITMAP); // to atomically read bitmap (__first_bit_set)
	      MPI_Type_commit(&MPI_T_BITMAP);
	      if (internal::MPI_3INT == MPI_DATATYPE_NULL)
		      MPI_Type_contiguous(3, MPI_INT, &internal::MPI_3INT), MPI_Type_commit(&internal::MPI_3INT);
	      
	      // update producer count stored in the controller
	      MPI_Win_lock(MPI_LOCK_SHARED, rank_controller[0], 0, w_ctl);
	      if (role != mpi_mpmc_role::CONSUMER) {
		      unsigned origin = 1, ignored;
		      MPI_Fetch_and_op(&origin, &ignored, MPI_INT, rank_controller[0],
				      _offset_prod_count, MPI_SUM, w_ctl);
	      }
	      MPI_Win_unlock(rank_controller[0], w_ctl);

	      MPI_Barrier(comm);
	      { // this queue has been enabled; wake up any thread that called push()/pop()
		      std::lock_guard<std::mutex> lk{m};
		      enabled.store(true);
		      cv.notify_all();
	      }
      }

      bool is_empty() const noexcept;
      T pop();
      bool push(T item);

   private:
      std::tuple<unsigned, unsigned> write_begin(int rank, size_t item_size,
						 unsigned offset_ipw_pr,
						 size_t buffer_size) noexcept;
      void write_end(int rank, size_t item_size, unsigned offset_pw, unsigned current) noexcept;

      std::tuple<unsigned, unsigned> read_begin(int rank, size_t item_size,
						unsigned offset_ipr_pw) noexcept;
      void read_end(int rank, size_t item_size, unsigned offset_pr, unsigned current) noexcept;

      T pop_data(int rank, unsigned tail, size_t length);
      std::tuple<unsigned, size_t> push_data(T item);
      std::tuple<int, unsigned, size_t> pop_metadata();
      void push_metadata(int rank, std::tuple<unsigned, size_t> loc_length);
      void do_producer_eos();

      void __set_bit(int nr, unsigned offset) {
	      unsigned mask = 1 << (nr & 0x1f), ignored;
	      MPI_Fetch_and_op(&mask, &ignored, MPI_INT, rank_controller[0],
			      offset + (nr >> 5), MPI_BOR, w_ctl);
	      MPI_Win_flush(rank_controller[0], w_ctl);
      }
      bool __clear_bit_CAS(int nr, unsigned offset, unsigned cmp) {
	      unsigned masked = cmp & ~(1 << (nr & 0x1f)), orig;
	      MPI_Compare_and_swap(&masked, &cmp, &orig, MPI_INT,
				   rank_controller[0], offset + (nr >> 5), w_ctl);
	      MPI_Win_flush(rank_controller[0], w_ctl);
	      return orig == cmp;
      }
      /**
        \brief find first bit set in a bitmap.
        \return A std::tuple<int, unsigned> that includes the position of
         the bit and the value of the MPI_INT that contained it.
	*/
      std::tuple<int, unsigned> __first_bit_set(unsigned offset, size_t len) {
	      unsigned ignored[len], b[len];
	      MPI_Fetch_and_op(ignored, b, MPI_T_BITMAP, rank_controller[0],
			       offset, MPI_NO_OP, w_ctl);
	      MPI_Win_flush(rank_controller[0], w_ctl);
	      for (unsigned i = 0, shl = 0; i < len; ++i, shl += sizeof(int) << 3)
		      if (b[i]) return std::make_tuple(shl + ::ffs(b[i])-1,
						       b[i]);
	      return {};
      }

      /**
        \brief This function was adapted from include/linux/circ_buf.h. 
         Return available octets in the circular buffer (or available up to the
         end of the buffer, whatever comes first).
        */
      unsigned int __circ_distance(unsigned ptr, size_t size) const noexcept
      { return size - ptr; }

      size_t bitmap_size() { return ((n_members - 1) >> 5) + 1; }
      size_t bitmap_size_octets() { return bitmap_size() * sizeof(int); }
      size_t get_ctl_region_size() {
	      if (role == mpi_mpmc_role::CONSUMER)
		      return 0;

	      size_t ret = (4 * sizeof(int)); /* data_internal_pwrite, data_pread, data_internal_pread, data_pwrite */
	      if (role == mpi_mpmc_role::CONTROLLER)
		      ret += sizeof(int)                                     /* producer count */
			      + (4 * sizeof(int)) + ((capacity * 3) * sizeof(int)) /* queue of metadata (rank + length) */
			      + (BITMAP_COUNT * bitmap_size_octets());                  /* role + wait bitmaps */
	      return ret;
      }

      enum __wait_bitmap { ROLE_BITMAP = 0, PRODUCER_BITMAP, CONSUMER_BITMAP,
		            BITMAP_COUNT };
      enum __wakeup_status { EOS = 0, RESOURCE_AVAILABLE };
      bool wait(__wait_bitmap i);
      void wake_up(__wait_bitmap i, __wakeup_status s = RESOURCE_AVAILABLE);

      int capacity, data_size, n_members = 0;
      bool is_consumer = false, eos_received = false, eos_sent = false;
      mpi_mpmc_role role;
      char *data, *ctl;        // producer-only
      // offset -in sizeof(int) units- of members of the control region
      constexpr static int _offset_data_internal_pwrite = 0, _offset_data_pread = 1,
		_offset_data_internal_pread = 2, _offset_data_pwrite = 3,
		// controller-only
		_offset_meta_internal_pwrite = 4, _offset_meta_pread = 5, // Infiniband RDMA fetch-and-op requires
									  // this to be 64-bit aligned
		_offset_meta_internal_pread = 6, _offset_meta_pwrite = 7,
		_offset_prod_count = 8,
		_offset_meta_array = 9;
      int _offset_bitmap[BITMAP_COUNT];

      // for mpi_mpmc_queue<T>::enable()
      std::atomic_bool        enabled{false};
      std::mutex              m{};
      std::mutex              r_m{}, w_m{}; // no more than one concurrent consumer/producer per proc
      std::condition_variable cv{};
      void block_if_not_enabled() noexcept {
	      std::unique_lock<std::mutex> lk{m};
	      if (!enabled.load()) cv.wait(lk);
      }
      
      void register_consumer() noexcept {
	      if (!is_consumer) {
		      __set_bit(my_rank, _offset_bitmap[0]); // role bitmap, 1=consumer
		      is_consumer = true;
	      }
      }
      
      // MPI
      MPI_Comm         comm;
      MPI_Win          w_data, w_ctl;
      MPI_Datatype     MPI_T_BITMAP;
      int              my_rank, rank_controller[1]; /* rank of the controller process; before enable(),
						     * it refers to MPI_COMM_WORLD; after enable(), it
						     * refers to the `comm' communicator */
};

/**
  \brief Add this proc to the wait bitmap `i`. This call blocks.
  \return `false` if EOS was reached; `true` otherwise.
  */
template <typename T>
bool mpi_mpmc_queue<T>::wait(__wait_bitmap i) {
	__set_bit(my_rank, _offset_bitmap[i]);
	// THREAD-SAFETY: this should probably be wrapped in #ifdef-#endif
	MPI_Win_unlock(rank_controller[0], w_ctl),
		internal::mutex_mpi.unlock();

	int buf;
	MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, i, comm, MPI_STATUS_IGNORE);

	internal::mutex_mpi.lock(),
		MPI_Win_lock(MPI_LOCK_SHARED, rank_controller[0], 0, w_ctl);
	return (buf != EOS);
}

/**
  \brief Wake up one proc from the wait bitmap `i`.
  */
template <typename T>
void mpi_mpmc_queue<T>::wake_up(__wait_bitmap i, __wakeup_status s) {
	std::tuple<int, unsigned> t;
	do {
		t = __first_bit_set(_offset_bitmap[i], bitmap_size());
		if (!std::get<1>(t)) // nothing to do
			return;
	} while (!__clear_bit_CAS(std::get<0>(t), _offset_bitmap[i], std::get<1>(t)));
	MPI_Send(&s, 1, MPI_INT, std::get<0>(t), i, comm);
}

template <typename T>
bool mpi_mpmc_queue<T>::is_empty() const noexcept {
	unsigned ignored, pread, pwrite;
	MPI_Fetch_and_op(&ignored, &pread, MPI_INT, rank_controller[0], _offset_meta_pread,
			MPI_NO_OP, w_ctl);
	MPI_Fetch_and_op(&ignored, &pwrite, MPI_INT, rank_controller[0], _offset_meta_pwrite,
			MPI_NO_OP, w_ctl);
	MPI_Win_flush(rank_controller[0], w_ctl);
	return pread == pwrite;
}

/**
  \brief Begin a read operation on a circular buffer, regardless of the buffer type.
   Sleeps if the buffer is empty.
  \param rank Rank of the process that owns the circular buffer.
  \param item_size Length of the item to read
  \param offset_ipr_pw Offset in w_ctl of the internal_pread-pwrite pair
  \return A std::tuple including current [head,tail] pointers
  */
template <typename T>
std::tuple<unsigned, unsigned> mpi_mpmc_queue<T>::read_begin(int rank, size_t item_size,
							     unsigned offset_ipr_pw) noexcept {
	unsigned ignored[2], p[2], cas[2];

	do {
		do {
			// HACK: see write_begin()
			MPI_Fetch_and_op(ignored, p, MPI_2INT, rank,
					offset_ipr_pw, MPI_NO_OP, w_ctl);
			MPI_Win_flush(rank, w_ctl);
			if (p[0]/*internal_pread*/ >= p[1]/*pwrite*/) // empty queue
				if (!wait(CONSUMER_BITMAP))
					return std::make_tuple(0U, 0U); // EOS-terminated wait()
		} while (p[0]/*internal_pread*/ >= p[1]/*pwrite*/);

		cas[0] = p[0]/*internal_pread*/ + item_size;
		cas[1] = p[0];
		MPI_Compare_and_swap(&cas[0], &cas[1], &p[0], MPI_INT, rank,
				     offset_ipr_pw, w_ctl);
		MPI_Win_flush(rank, w_ctl);
	} while (cas[1] != p[0]);
	return std::make_tuple(p[1], p[0]);
}

/**
  \brief End a read operation on a circular buffer (updates pread). See
   begin_read() documentation.
  */
template <typename T>
void mpi_mpmc_queue<T>::read_end(int rank, size_t item_size, unsigned offset_pr,
				 unsigned current) noexcept {
	unsigned pread = current + item_size, tmp;
	do {
		MPI_Compare_and_swap(&pread, &current, &tmp, MPI_INT, rank,
				     offset_pr, w_ctl);
		MPI_Win_flush(rank, w_ctl);
	} while (tmp != current);
}

/**
  \brief Begin a write operation on a circular buffer, regardless of the buffer type.
   Sleeps if the buffer is full.
  \param rank Rank of the process that owns the circular buffer.
  \param item_size Length of the item to write
  \param offset_ipw_pr Offset in w_ctl of the internal_pwrite-pread pair
  \param buffer_size Length (in octets) of the buffer
  \return A std::tuple including current [head,tail] pointers
  */
template <typename T>
std::tuple<unsigned, unsigned> mpi_mpmc_queue<T>::write_begin(int rank, size_t item_size,
							      unsigned offset_ipw_pr,
							      size_t buffer_size) noexcept{
	unsigned ignored[2], p[2], cas[2];

	do {
		// HACK: this is required, as MPI_Compare_and_swap() does not
		// work properly (at least in openmpi 3.1.2-1) with MPI_2INT
		do {
			MPI_Fetch_and_op(ignored, p, MPI_2INT, rank,
					offset_ipw_pr, MPI_NO_OP, w_ctl);
			MPI_Win_flush(rank, w_ctl);
			if ((p[0]/*internal_pwrite*/ + item_size) >= (p[1]/*pread*/ + buffer_size)) // full queue
				wait(PRODUCER_BITMAP);
		} while ((p[0]/*internal_pwrite*/ + item_size) >= (p[1]/*pread*/ + buffer_size));

		cas[0] = p[0]/*internal_pwrite*/ + item_size;
		cas[1] = p[0];
		MPI_Compare_and_swap(&cas[0], &cas[1], &p[0], MPI_INT, rank,
				     offset_ipw_pr, w_ctl);
		MPI_Win_flush(rank, w_ctl);
	} while (cas[1] != p[0]);
	return std::make_tuple(p[0], p[1]);
}

/**
  \brief End a write operation on a circular buffer (updates pwrite). See
   begin_write() documentation.
  */
template <typename T>
void mpi_mpmc_queue<T>::write_end(int rank, size_t item_size, unsigned offset_pw,
				  unsigned current) noexcept {
	unsigned pwrite = current + item_size, tmp;
	do {
		MPI_Compare_and_swap(&pwrite, &current, &tmp, MPI_INT, rank,
				     offset_pw, w_ctl);
		MPI_Win_flush(rank, w_ctl);
	} while (tmp != current);
}


template <typename T>
T mpi_mpmc_queue<T>::pop(){
	block_if_not_enabled();

	{
		std::lock(r_m, internal::mutex_mpi); // no scoped_lock in C++14
		std::lock_guard<std::mutex> lk1{r_m, std::adopt_lock};
		std::lock_guard<std::mutex> lk2{internal::mutex_mpi, std::adopt_lock};

		if (eos_received)
			return {};    // EOS

		// TODO: make a RAII object for MPI_Win_lock()/MPI_Win_unlock() calls
		MPI_Win_lock(MPI_LOCK_SHARED, rank_controller[0], 0, w_ctl);
		register_consumer();
		auto meta = pop_metadata();
		if (0 == std::get<2>(meta) /*length*/) {
			MPI_Win_unlock(rank_controller[0], w_ctl);
			eos_received = true; 
			return {};    // EOS
		}

		auto item = pop_data(std::get<0>(meta), std::get<1>(meta), std::get<2>(meta));
		wake_up(PRODUCER_BITMAP);
		MPI_Win_unlock(rank_controller[0], w_ctl);
		return std::move(item);
	}
}

template <typename T>
bool mpi_mpmc_queue<T>::push(T item){
	block_if_not_enabled();
       
	// A push on a consumer-only queue is ignored (never fails)
	if (role == mpi_mpmc_role::CONSUMER || eos_sent) return true;
	{
		std::lock(w_m, internal::mutex_mpi); // no scoped_lock in C++14
		std::lock_guard<std::mutex> lk1{w_m, std::adopt_lock};
		std::lock_guard<std::mutex> lk2{internal::mutex_mpi, std::adopt_lock};

		MPI_Win_lock(MPI_LOCK_SHARED, rank_controller[0], 0, w_ctl);
		if (item.first) {
			push_metadata(my_rank,
					push_data(item));
			wake_up(CONSUMER_BITMAP);
		} else {  // EOS
			do_producer_eos();
		}
		MPI_Win_unlock(rank_controller[0], w_ctl);
	}
	return true;    // not required as of 5e7418c6 (May 25 2018)
}

template <typename T>
void mpi_mpmc_queue<T>::do_producer_eos() {
	unsigned origin = -1, prod_count;
	MPI_Fetch_and_op(&origin, &prod_count, MPI_INT, rank_controller[0],
			_offset_prod_count, MPI_SUM, w_ctl);
	MPI_Win_flush(rank_controller[0], w_ctl);
	eos_sent = true;

	if (prod_count != 1) // other producers still available
		return;

	// FIXME: quick-and-dirty way to send EOS to consumers
	unsigned len = bitmap_size();
	unsigned ignored[len], b[len];
	MPI_Fetch_and_op(ignored, b, MPI_T_BITMAP, rank_controller[0],
			_offset_bitmap[0], MPI_NO_OP, w_ctl);
	MPI_Win_flush(rank_controller[0], w_ctl);
	__wakeup_status s[] = { RESOURCE_AVAILABLE, EOS };
	for (unsigned i = 0, shl = 0; i < len; ++i, shl += sizeof(int) << 3) {
		for (unsigned nr = ::ffs(b[i]); b[i];
				b[i] >>= nr, shl += nr, nr = ::ffs(b[i])) {
			auto rank = (shl + nr)-1;
			MPI_Send(&s[0], 1, MPI_INT, rank, CONSUMER_BITMAP, comm),
				MPI_Send(&s[1], 1, MPI_INT, rank, CONSUMER_BITMAP, comm);
		}
	}
}

/**
  \brief Fetches a T instance of `length` octets located at `rank` and
   deserializes it. 
  */
template <typename T>
T mpi_mpmc_queue<T>::pop_data(int rank, unsigned tail, size_t length) {
	if (rank_controller[0] != rank) MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, w_ctl);
	//std::tie(head, tail) = read_begin(rank, length, _offset_data_internal_pread);

	// RMA-copy serialized object from target process `rank`
	std::string buf(length, 0);
	MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, w_data);
	for (unsigned current_tail = tail, offset = 0, count = length, c = 0; count > 0;
	     offset += c, count -= c) {
		current_tail = (current_tail +c)%data_size;
		c = __circ_distance(current_tail, data_size);
		if (count < c)
			c = count;

		MPI_Get(const_cast<std::string::value_type *>(buf.data()) +offset, c, MPI_BYTE,
				rank, current_tail, c, MPI_BYTE, w_data);
	}
	MPI_Win_unlock(rank, w_data);

	read_end(rank, length, _offset_data_pread, tail);
	if (rank_controller[0] != rank) MPI_Win_unlock(rank, w_ctl);

	std::stringbuf sbuf{buf};
	iarchive_type ia{sbuf};
	T data;
	ia >> data;
	return std::move(data);
}

/**
  \brief Serializes and copies an object to the data buffer of this producer.
  \return Size in octets of serialized object.
  */
template <typename T>
std::tuple<unsigned, size_t> mpi_mpmc_queue<T>::push_data(T item) {
	std::ostringstream os{};
	oarchive_type      oa{os};
	oa << item;

	const std::string &buf = os.str();
	if (rank_controller[0] != my_rank) MPI_Win_lock(MPI_LOCK_SHARED, my_rank, 0, w_ctl);
	unsigned int head, tail;
	std::tie(head, tail) = write_begin(my_rank, buf.size(), _offset_data_internal_pwrite,
					   data_size);

	// copy serialized object to the data buffer. MPI_LOCK_SHARED should work here
	MPI_Win_lock(MPI_LOCK_SHARED, my_rank, 0, w_data);
	for (unsigned current_head = head, offset = 0, count = buf.size(), c = 0; count > 0;
	     offset += c, count -= c) {
		current_head = (current_head +c)%data_size;
		c = __circ_distance(current_head, data_size);
		if (count < c)
			c = count;

		MPI_Put(buf.data() +offset, c, MPI_BYTE, my_rank, current_head, c, MPI_BYTE, w_data);
	}
	MPI_Win_unlock(my_rank, w_data);

	write_end(my_rank, buf.size(), _offset_data_pwrite, head);
	if (rank_controller[0] != my_rank) MPI_Win_unlock(my_rank, w_ctl);

	return std::make_tuple(head, buf.size());
}

/**
  \brief Get metadata (rank + item location + length) of the next item in the queue.
  \return A std::tuple<int, unsigned, size_t> providing the location and length of the item.
  */
template <typename T>
std::tuple<int, unsigned, size_t> mpi_mpmc_queue<T>::pop_metadata() {
	unsigned circbuf_len = (capacity * 3), head, tail;
	std::tie(head, tail) = read_begin(rank_controller[0], 3, _offset_meta_internal_pread);
	if (tail == head)    // EOS
		return std::make_tuple(0, 0U, 0U);

	int triplet[3], tmp[3];
	MPI_Fetch_and_op(tmp, triplet, internal::MPI_3INT, rank_controller[0],
			_offset_meta_array +(tail%circbuf_len), MPI_NO_OP, w_ctl);

	read_end(rank_controller[0], 3, _offset_meta_pread, tail);
	return std::make_tuple(triplet[0], triplet[1], triplet[2]);
}

/**
  \brief Make new data available in the queue. The new element has `length`-bytes and is
  located at `rank`.
  */
template <typename T>
void mpi_mpmc_queue<T>::push_metadata(int rank, std::tuple<unsigned, size_t> loc_length) {
	unsigned circbuf_len = (capacity * 3), head, tail;
	std::tie(head, tail) = write_begin(rank_controller[0], 3,
					   _offset_meta_internal_pwrite, circbuf_len);

	unsigned triplet[] = { (unsigned)rank, (unsigned)std::get<0>(loc_length),
				(unsigned)std::get<1>(loc_length) }, tmp[3];
	MPI_Fetch_and_op(triplet, tmp, internal::MPI_3INT, rank_controller[0],
			_offset_meta_array +(head%circbuf_len), MPI_REPLACE, w_ctl);

	write_end(rank_controller[0], 3, _offset_meta_pwrite, head);
}

template <>
class mpi_mpmc_queue<void>{
   public:
      using value_type = void;
};

namespace internal {
template <typename T>
struct is_queue<mpi_mpmc_queue<T>> : std::true_type {};
}

}
#endif
