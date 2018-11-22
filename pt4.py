from math import radians, sin, asin, sqrt, cos


def haversine(truck_latitude, truck_longitude, customer_latitude, customer_longitude):
    radius = 2.5  # In Kilometer
    # Degree to radians
    truck_latitude, truck_longitude, customer_latitude, customer_longitude = \
        map(radians, [truck_latitude, truck_longitude, customer_latitude, customer_longitude])

    # Haversine formula
    d_latitude = customer_latitude - truck_latitude
    d_longitude = customer_longitude - truck_longitude
    a = sin(d_latitude / 2) ** 2 + cos(truck_latitude) * cos(customer_latitude) * sin(d_longitude / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    print(c * r)
    return c * r <= radius


print(haversine(43.0482894, -76.1203977, 43.0438258, -76.1458835))
