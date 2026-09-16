#include "plat.h"

#include <windows.h>
#include <cfgmgr32.h>
#include <setupapi.h>
#include <winioctl.h>

#define AIMDO_DISK_UNKNOWN (-1)
#define AIMDO_DISK_SLOW      0
#define AIMDO_DISK_FAST      1

#define BUS_TYPE_UNKNOWN             0
#define BUS_TYPE_USB                 7
#define BUS_TYPE_RAID                8
#define BUS_TYPE_ISCSI               9
#define BUS_TYPE_VIRTUAL            14
#define BUS_TYPE_FILE_BACKED_VIRTUAL 15
#define BUS_TYPE_SPACES             16
#define BUS_TYPE_NVME               17
#define BUS_TYPE_NVMEOF             20
#define BUS_TYPE_MAX                21

static const GUID disk_interface_guid = {
    0x53f56307, 0xb6bf, 0x11d0, {0x94, 0xf2, 0x00, 0xa0, 0xc9, 0x1e, 0xfb, 0x8b}
};
static const GUID pci_property_guid = {
    0x3ab22e31, 0x8264, 0x4b4e, {0x9a, 0xf5, 0xa8, 0xd2, 0xd8, 0xe3, 0x3e, 0x62}
};

static HANDLE open_existing(const wchar_t *path) {
    return CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                       NULL, OPEN_EXISTING, 0, NULL);
}

static bool get_volume_guid(const wchar_t *path, wchar_t volume[64]) {
    wchar_t buffer[32768];
    wchar_t *end;
    DWORD length;
    HANDLE file = open_existing(path);

    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    length = GetFinalPathNameByHandleW(file, buffer, ARRAY_SIZE(buffer),
                                       FILE_NAME_NORMALIZED | VOLUME_NAME_GUID);
    CloseHandle(file);
    if (!length || length >= ARRAY_SIZE(buffer) || wcsncmp(buffer, L"\\\\?\\Volume{", 11) ||
        !(end = wcschr(buffer + 11, L'}')) || end - buffer + 2 >= 64) {
        return false;
    }
    end[1] = L'\\';
    end[2] = L'\0';
    wcscpy_s(volume, 64, buffer);
    return true;
}

static int disk_bus_type(DWORD disk_number) {
    STORAGE_PROPERTY_QUERY query = {
        .PropertyId = StorageDeviceProperty,
        .QueryType = PropertyStandardQuery,
    };
    STORAGE_DEVICE_DESCRIPTOR descriptor;
    wchar_t path[64];
    DWORD returned;
    HANDLE disk;

    swprintf_s(path, ARRAY_SIZE(path), L"\\\\.\\PhysicalDrive%lu", disk_number);
    disk = open_existing(path);
    if (disk == INVALID_HANDLE_VALUE) {
        return -1;
    }
    if (!DeviceIoControl(disk, IOCTL_STORAGE_QUERY_PROPERTY,
                         &query, sizeof(query), &descriptor, sizeof(descriptor),
                         &returned, NULL) || returned < offsetof(STORAGE_DEVICE_DESCRIPTOR, RawDeviceProperties)) {
        CloseHandle(disk);
        return -1;
    }
    CloseHandle(disk);
    return descriptor.BusType;
}

static DEVINST disk_devinst(DWORD disk_number) {
    SP_DEVICE_INTERFACE_DATA interface_data = {0};
    SP_DEVINFO_DATA devinfo = {0};
    HDEVINFO devices;
    DEVINST found = 0;

    devices = SetupDiGetClassDevsW(&disk_interface_guid, NULL, NULL,
                                   DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (devices == INVALID_HANDLE_VALUE) {
        return 0;
    }
    interface_data.cbSize = sizeof(interface_data);
    for (DWORD index = 0; SetupDiEnumDeviceInterfaces(
             devices, NULL, &disk_interface_guid, index, &interface_data); index++) {
        PSP_DEVICE_INTERFACE_DETAIL_DATA_W detail;
        STORAGE_DEVICE_NUMBER number;
        DWORD required = 0;
        DWORD returned;
        HANDLE disk;

        SetupDiGetDeviceInterfaceDetailW(devices, &interface_data, NULL, 0, &required, NULL);
        if (required < sizeof(*detail) || !(detail = malloc(required))) {
            continue;
        }
        detail->cbSize = sizeof(*detail);
        devinfo.cbSize = sizeof(devinfo);
        if (!SetupDiGetDeviceInterfaceDetailW(devices, &interface_data, detail, required,
                                              NULL, &devinfo) ||
            (disk = open_existing(detail->DevicePath)) == INVALID_HANDLE_VALUE) {
            free(detail);
            continue;
        }
        if (DeviceIoControl(disk, IOCTL_STORAGE_GET_DEVICE_NUMBER, NULL, 0,
                            &number, sizeof(number), &returned, NULL) &&
            returned >= sizeof(number) && number.DeviceType == FILE_DEVICE_DISK &&
            number.DeviceNumber == disk_number) {
            found = devinfo.DevInst;
        }
        CloseHandle(disk);
        free(detail);
        if (found) {
            break;
        }
    }
    SetupDiDestroyDeviceInfoList(devices);
    return found;
}

static bool pci_property(DEVINST devinst, DWORD pid, DWORD *value) {
    DEVPROPKEY key = {.fmtid = {0}, .pid = pid};
    DEVPROPTYPE type;
    ULONG size = sizeof(*value);

    key.fmtid = pci_property_guid;
    return CM_Get_DevNode_PropertyW(devinst, &key, &type, (PBYTE)value, &size, 0) == CR_SUCCESS &&
           type == DEVPROP_TYPE_UINT32 && size == sizeof(*value);
}

static DEVINST nvme_controller(DEVINST devinst) {
    for (int depth = 0; depth < 16; depth++) {
        DWORD base_class;
        DWORD sub_class;
        DWORD prog_if;
        DEVINST parent;

        if (pci_property(devinst, 3, &base_class) && base_class == 0x01 &&
            pci_property(devinst, 4, &sub_class) && sub_class == 0x08 &&
            pci_property(devinst, 5, &prog_if) && prog_if == 0x02) {
            return devinst;
        }
        if (CM_Get_Parent(&parent, devinst, 0) != CR_SUCCESS) {
            break;
        }
        devinst = parent;
    }
    return 0;
}

static int physical_disk_fast(DWORD disk_number) {
    int bus_type = disk_bus_type(disk_number);
    DEVINST devinst;
    DEVINST controller;
    DWORD generation;
    DWORD width;

    if (bus_type != BUS_TYPE_NVME) {
        if (bus_type <= BUS_TYPE_UNKNOWN || bus_type >= BUS_TYPE_MAX ||
            bus_type == BUS_TYPE_USB || bus_type == BUS_TYPE_RAID ||
            bus_type == BUS_TYPE_ISCSI || bus_type == BUS_TYPE_VIRTUAL ||
            bus_type == BUS_TYPE_FILE_BACKED_VIRTUAL || bus_type == BUS_TYPE_SPACES ||
            bus_type == BUS_TYPE_NVMEOF) {
            return AIMDO_DISK_UNKNOWN;
        }
        return AIMDO_DISK_SLOW;
    }
    devinst = disk_devinst(disk_number);
    controller = devinst ? nvme_controller(devinst) : 0;
    if (!controller || !pci_property(controller, 9, &generation) ||
        !pci_property(controller, 10, &width)) {
        return AIMDO_DISK_UNKNOWN;
    }
    log(DEBUG, "%s: disk=%lu current_gen=%lu current_width=%lu\n",
        __func__, disk_number, generation, width);

    return ((generation == 3 && width == 4) ||
            (generation == 4 && width == 4) ||
            (generation == 5 && (width == 2 || width == 4))) ?
           AIMDO_DISK_FAST : AIMDO_DISK_SLOW;
}

static int volume_fast(const wchar_t *volume) {
    union {
        VOLUME_DISK_EXTENTS extents;
        BYTE bytes[32768];
    } buffer;
    VOLUME_DISK_EXTENTS *extents = &buffer.extents;
    wchar_t volume_path[64];
    DWORD returned;
    DWORD count;
    HANDLE handle;
    int result = AIMDO_DISK_FAST;

    wcscpy_s(volume_path, ARRAY_SIZE(volume_path), volume);
    volume_path[wcslen(volume_path) - 1] = L'\0';
    handle = open_existing(volume_path);
    if (handle == INVALID_HANDLE_VALUE) {
        return AIMDO_DISK_UNKNOWN;
    }
    bool success = DeviceIoControl(handle, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
                                   NULL, 0, extents, sizeof(buffer), &returned, NULL);
    CloseHandle(handle);
    if (!success || returned < offsetof(VOLUME_DISK_EXTENTS, Extents) ||
        returned > sizeof(buffer)) {
        return AIMDO_DISK_UNKNOWN;
    }
    count = extents->NumberOfDiskExtents;
    if (!count || count > (returned - offsetof(VOLUME_DISK_EXTENTS, Extents)) /
                          sizeof(extents->Extents[0])) {
        return AIMDO_DISK_UNKNOWN;
    }
    for (DWORD i = 0; i < count; i++) {
        int disk_result = physical_disk_fast(extents->Extents[i].DiskNumber);

        if (disk_result == AIMDO_DISK_SLOW) {
            return AIMDO_DISK_SLOW;
        }
        if (disk_result == AIMDO_DISK_UNKNOWN) {
            result = AIMDO_DISK_UNKNOWN;
        }
    }
    return result;
}

SHARED_EXPORT
int aimdo_storage_fast_disk(const wchar_t *path) {
    wchar_t volume[64];

    if (!path || !get_volume_guid(path, volume)) {
        return AIMDO_DISK_UNKNOWN;
    }
    return volume_fast(volume);
}
