/*
*	Copyright (C) 2026 openEMS contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "temporal_blocking.h"

#include <cctype>
#include <cstdlib>

static bool ReadCount(const std::string& text, unsigned int& value)
{
	if (text.empty() || text.size() > 6)
		return false;
	for (char c : text)
		if (!std::isdigit((unsigned char)c))
			return false;
	value = (unsigned int)std::strtoul(text.c_str(), nullptr, 10);
	return true;
}

bool TemporalBlockingRequest::Parse(const std::string& text, TemporalBlockingRequest& req)
{
	if (text == "auto")
	{
		req.mode = AUTO;
		req.k = req.W = 0;
		return true;
	}
	if (text == "off" || text == "0")
	{
		req.mode = OFF;
		req.k = req.W = 0;
		return true;
	}

	const size_t colon = text.find(':');
	unsigned int k = 0, W = 0;
	if (!ReadCount(text.substr(0, colon), k) || k == 0)
		return false;
	if (colon != std::string::npos && (!ReadCount(text.substr(colon + 1), W) || W == 0))
		return false;
	req.mode = FORCED;
	req.k = k;
	req.W = W;
	return true;
}

std::string TemporalBlockingRequest::ToString() const
{
	switch (mode)
	{
	case AUTO:
		return "auto";
	case OFF:
		return "off";
	case FORCED:
		break;
	}
	return W ? std::to_string(k) + ":" + std::to_string(W) : std::to_string(k);
}

TemporalBlockingRequest TemporalBlockingRequest::Off()
{
	TemporalBlockingRequest req;
	req.mode = OFF;
	return req;
}

TemporalBlockingRequest TemporalBlockingRequest::Forced(unsigned int k, unsigned int W)
{
	TemporalBlockingRequest req;
	req.mode = FORCED;
	req.k = k;
	req.W = W;
	return req;
}
