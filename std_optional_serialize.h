/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// Provides serialization for std::optional (serialization forked from boost::optional)

#ifndef STD_OPTIONAL_SERIALIZE_H_
#define STD_OPTIONAL_SERIALIZE_H_

#include <boost/archive/detail/basic_iarchive.hpp>
#include <experimental/optional>

#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>

namespace std {
	template <typename T>
	using optional = experimental::optional<T>; // TODO: protect using #ifdef __cplusplus
}
// function specializations must be defined in the appropriate namespace - boost::serialization
namespace boost { 
namespace serialization {

template<class Archive, class T>
void save(Archive& ar, const std::optional<T>& t, const unsigned int /*version*/) {
	const bool has_value = t;
	ar << boost::serialization::make_nvp("has_value", has_value);
	if (has_value)
		ar << boost::serialization::make_nvp("value", t.value());
}

template<class Archive, class T>
void load(Archive& ar, std::optional<T>& t, const unsigned int version) {
	bool has_value;
	ar >> boost::serialization::make_nvp("has_value", has_value);
	if (!has_value) {
		t = T();
		return;
	}

	if (0 == version) {
		boost::serialization::item_version_type item_version(0);
		boost::archive::library_version_type    library_version(
				ar.get_library_version()
				);
		if (boost::archive::library_version_type(3) < library_version)
			ar >> BOOST_SERIALIZATION_NVP(item_version);
	}
	if (has_value)
		t = T();
	ar >> boost::serialization::make_nvp("value", t.value());
}

template<class Archive, class T>
void serialize(Archive& ar, std::optional<T>& t, const unsigned int version) {
	boost::serialization::split_free(ar, t, version);
}

template<class T>
struct version<std::optional<T> > {
	BOOST_STATIC_CONSTANT(int, value = 1);
};

} // serialization
} // boost
#endif // STD_OPTIONAL_SERIALIZE_H_
